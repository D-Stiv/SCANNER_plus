import numpy as np
from config import Configuration, setup_config
from util import set_seed
from loader import load_dataset
import torch
from trainer import trainer
from metrics import get_metrics
import time

     
config = Configuration()
setup_config(config)

exp_num = 1
tot_exp = len(config.get_configs())
for args in config.get_configs():
    set_seed(args.seed)
    print(f'Starting experiment number {exp_num}/{tot_exp} ...')
    
    args.expnum = exp_num if args.expid < 0 else args.expnum

    args.expid = np.random.randint(10, 10000) if args.expid < 0 else args.expid
    if not (args.use_neighbors and args.num_neighbors > 0):
        args.num_neighbors = args.temporal_horizons = args.distance = args.num_heads = None
 
    dataloader = load_dataset(
            args=args, dataset_name=args.dataset_name, input_horizon=args.input_horizon, output_horizon=args.output_horizon,
            batch_size=args.batch_size, start_hour=args.start_hour, end_hour=args.end_hour, num_spatial=args.distance
        )
    scaler = dataloader['scaler']
    
    args.device = device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    adj_mx = None
    supports = None if adj_mx is None else [torch.tensor(i).to(device) for i in adj_mx]


    if args.adaptadj or supports is None:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    # Get Shape of Data
    _, _, num_nodes, in_feats = dataloader['x_train'].shape  # _ is seq_length
    
    in_dim = in_feats - 1

    engine = trainer(args=args, device=device, scaler=scaler, num_nodes=num_nodes, adjinit=adjinit, in_dim=in_dim, supports=supports, num_spatial=args.distance)
    save_dir = f'[Path to the model checkpoint directory]/{args.dataset_name}/{args.model_name}'
    
    train_results = engine.train_(dataloader=dataloader, save_dir=save_dir)

    phase = 'test'
    checkpoint = torch.load(f'{save_dir}/_id{args.expid}_num{args.expnum}_best_.pth')
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.model.to(device)
    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    engine.nfe.load_state_dict(checkpoint['nfe_state_dict'])
    engine.nfe.to(device)

    t0 = time.time()
    amae, amape, armse = get_metrics(args=args, device=device, dataloader=dataloader, engine=engine, scaler=scaler, phase=phase)
    t1 = time.time()
    
    mean_amae = np.mean(amae)
    mean_amape = np.mean(amape)
    mean_armse = np.mean(armse)
    
    log = '{} - dataset: {}. On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(args.model_name, args.dataset_name, args.output_horizon, mean_amae, mean_amape, mean_armse))

    exp_num += 1
