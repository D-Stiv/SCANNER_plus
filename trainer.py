from metrics import masked_mae, masked_mape, masked_rmse
from models import gwnet, NFE
from correlations import get_correlations, aggregate_correlations, get_keys
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import numpy as np
import time
from util import mkdir
from metrics import get_metrics



class trainer():
    def __init__(self, args, device, scaler, num_nodes, adjinit, in_dim, supports, num_spatial=1):
        self.args = args
        self.device = device
        if args.use_neighbors:
            kernel_size = int((args.input_horizon + args.num_neighbors-1)/12) + 2
        else:
            kernel_size = int((args.input_horizon-1)/12) + 2
        self.model = gwnet(
                device=device, args=args, num_nodes=num_nodes, adjinit=adjinit, in_dim=in_dim, supports=supports, 
                output_horizon=args.output_horizon, nhid=args.nhid, dropout=args.dropout, gcn_bool=bool(args.gcn_bool), 
                snorm_bool=bool(args.snorm), tnorm_bool=bool(args.tnorm), adaptadj=args.adaptadj, kernel_size=kernel_size
           )
        self.model.to(device)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = 5
        if args.use_neighbors:
            not_learned_keys = None
            if args.corr_learn_type in ['dot_product', 'sum']:
                correlations = get_correlations(args)
                corrs = [np.concatenate([correlations[..., [i]], correlations[..., -num_spatial:]], axis=-1) for i in range(args.temporal_horizons)]
                learned_corr = np.stack([aggregate_correlations(corrs[i], args.corr_learn_type) for i in range(args.temporal_horizons)], axis=-1)
                not_learned_keys = get_keys(learned_corr, args.num_neighbors) if args.num_neighbors < correlations.shape[1] else learned_corr 
                not_learned_keys = torch.Tensor(not_learned_keys).to(device)
        
            self.nfe = NFE(args, args.num_heads, args.num_neighbors, num_nodes, args.input_horizon, in_feats=in_dim, temporal_horizons=args.temporal_horizons,
                           corr_learn_type=args.corr_learn_type, filter_agg_type=args.filter_agg_type, values_equal_keys=args.values_equal_keys,
                           num_spatial=num_spatial, not_learned_keys=not_learned_keys).to(device)
            params = [
                    {'params': self.model.parameters()},
                    {'params': self.nfe.parameters()}
                ]
        else:
            params = [
                    {'params': self.model.parameters()},
                ]
        self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)


    def enrich_(self, input, V=None):
        # input shape (batch_size, in_feats, num_nodes, input_horizon)
        out = self.nfe(input, V)  
        out = torch.cat([input, out], dim=-1)
        return out
        
    def train(self, input, real_val, values):  # 
        # input = [batch_size, 2, num_nodes, 12]
        # real_value = [batch_size, num_nodes, 12]
        # values = [batch_size, num_nodes, num_neighbors, num_keys]
        self.model.train()
        if self.args.use_neighbors:
            self.nfe.train()
            
        self.optimizer.zero_grad()
        if self.args.use_neighbors:
            input = self.enrich_(input, values)

        input = nn.functional.pad(input, (1, 0, 0, 0))
        # output, _ = self.model(input)
        output, _ = self.model(input)
        output = output.transpose(1, 3)

        # output = [batch_size,1,num_nodes,12]
        real = torch.unsqueeze(real_val, dim=1)
        # real = [batch_size,1,num_nodes,12]
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)

        loss.backward()
        
        del values, input, real_val, output
        gc.collect()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        if self.args.use_neighbors:
            self.nfe.eval()
            
            input = self.enrich_(input)
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real)
        mape = masked_mape(predict, real).item()
        rmse = masked_rmse(predict, real).item()
        return loss.item(), mape, rmse
    
    
    
    def train_(self, dataloader, save_dir, print_every=50, patience=15):
        self.args.patience = patience
        args = self.args
        device = self.device
        
        mkdir(save_dir)
        print('saving directory at ' + save_dir)

        print("start training...", flush=True)
        his_loss = []
        val_time = []
        train_time = []
        curr_min = np.inf
        wait = 0
        for i in range(1, args.epochs + 1):
            try:
                if wait >= args.patience:
                    print(log, f'early stop at epoch: {i:04d}')
                    break
                
                # training
                phase = "train"

                train_loss = []
                train_mape = []
                train_rmse = []
                t1 = time.time()
                dataloader['train_loader'].shuffle()

                t_ = time.time()
                for iter, (x, y, v) in enumerate(dataloader['train_loader'].get_iterator()):
                    trainx = torch.Tensor(x[..., :-1]).to(device)   	# speed and time_of_day
                    trainx = trainx.transpose(1, 3)
                    trainy = torch.Tensor(y).to(device)
                    trainy = trainy.transpose(1, 3)
                    
                    trainv = torch.Tensor(v[..., :-1,:]).to(device)   	# speed and time_of_day
                    trainv = trainv.transpose(1, 3)
                    
                    del x, y, v

                    metrics = self.train(trainx, trainy[:, 0, :, :], trainv)  # 0: speed
                    
                    del trainx, trainy, trainv

                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])
                    if iter % print_every == 0:
                        log = '{} - dataset: {}. Iter: {:03d}, Train Loss: {:.4f}, Time: {:.2f} secs'
                        print(log.format(args.model_name, args.dataset_name, iter, train_loss[-1], (time.time()-t_)),
                            flush=True)
                        t_ = time.time()


                t2 = time.time()
                train_time.append(t2 - t1)

                mtrain_loss = np.mean(train_loss)

                # validation
                phase = "val"

                s1 = time.time()
                valid_loss, valid_mape, valid_rmse = get_metrics(args=args, device=device, dataloader=dataloader, engine=self, scaler=dataloader['scaler'], phase=phase, ep=i)
                s2 = time.time()
                log = '{} - dataset: {}. Epoch: {:03d}/{:03d}, Inference Time: {:.4f} secs'
                print(log.format(args.model_name, args.dataset_name, i, args.epochs,(s2 - s1)))
                val_time.append(s2 - s1)

                mvalid_loss = np.mean(valid_loss)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)

                his_loss.append(mvalid_loss)

                log = '{} - dataset: {}. Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}, Training Time: {:.1f} secs/epoch. Best epoch: {}, best val loss: {:.4f}'
                print(log.format(args.model_name, args.dataset_name, i, mtrain_loss, mvalid_loss, mvalid_mape,
                                mvalid_rmse, (t2 - t1), np.argmin(his_loss), np.min(his_loss)), flush=True)
                
                # save only if less than the minimum
                if mvalid_loss == np.min(his_loss):
                    wait = 0
                    print(f'{args.model_name} - dataset: {args.dataset_name}. Epoch: {i:03d} - Val_loss decreases from {curr_min:.4f} to {mvalid_loss:.4f}')
                    curr_min = mvalid_loss
                    if args.use_neighbors:
                        torch.save({'model_state_dict': self.model.state_dict(), 'nfe_state_dict': self.nfe.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 
                                   f'{save_dir}/_id{args.expid}_num{args.expnum}_best_.pth')
                    else:
                        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 
                               f'{save_dir}/_id{args.expid}_num{args.expnum}_best_.pth')
                elif abs(mvalid_loss - np.min(his_loss))/mvalid_loss < 1e-3: # we are still in the elbow zone
                    pass
                else:
                    wait += 1
            except:
                if i > 1:
                    break
                else:
                    raise Exception
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        bestid = np.argmin(his_loss)
        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
        
        train_results = {
            "train_time_epoch": np.mean(train_time),
            "val_time_epoch": np.mean(val_time),
            "best_epoch": bestid,
            "effecient_epoch": len(his_loss),
            "best_val_loss": his_loss[bestid],
            "his_loss": his_loss,
        }
        
        return train_results
        

        
