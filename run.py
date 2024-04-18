import torch
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import linalg
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import configargparse
import copy
import random


def set_seed(seed=42):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class SNorm(nn.Module):
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self, num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class gwnet(nn.Module):
    def __init__(self, num_nodes, in_dim, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        out_dim = output_horizon
        residual_channels = nhid
        dilation_channels = nhid
        skip_channels = nhid * 8
        end_channels = nhid * 16
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 0
        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                if self.tnorm_bool:
                    self.tn.append(TNorm(num_nodes, residual_channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(residual_channels))
                self.filter_convs.append(nn.Conv2d(in_channels=num * residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=num * residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):        
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            x_list = []
            x_list.append(x)
            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, skip


def get_correlations():
    # transform distances in correlations and put diagonal to zero.
    temp_path = 'location of the temporal correlation matrix B (LxNxN) saved in .pkl'
    
    corr = []

    # temporal correlation
    with open(temp_path, "rb") as f:
        temp_corr = pickle.load(f)

    temp_horizon = np.min([args.temporal_horizons, temp_corr.shape[0]])
    for i in range(temp_horizon):
        c_i = temp_corr[i]
        corr.append(c_i)
    
    # spatial distance
    spatial_path = 'location of the spatial similarity matrix A (NxN) saved in .pkl'
    with open(spatial_path, "rb") as f:
        dist = pickle.load(f)
    corr.append(dist)

    corr = np.stack(corr, axis=-1)
    return corr  
    

def get_keys(corr, num_neighbors):
    # corr shape (num_nodes, num_nodes, temporal_horizon)
    corr[corr==0] = 1e-10

    if isinstance(corr, torch.Tensor):    
        num_nodes = corr.shape[0]
        def get_key(corr_):
            corr_ = corr_.fill_diagonal_(0)
            key = []
            for j in range(num_nodes):
                vect = corr_[j]
                top = torch.argsort(vect, descending=True)[:num_neighbors]
                mask_j = torch.zeros(num_nodes).to(device)
                for t in top:
                    mask_j[t] = vect[t]
                key.append(mask_j)
            key = torch.stack(key, dim=0)
            return key        
        keys = torch.stack([get_key(corr[..., i]) for i in range(corr.shape[-1])], dim=-1)
    else: # numpy array
        num_nodes = corr.shape[0]
        def get_key(corr_):
            np.fill_diagonal(corr_, 0)
            key = []
            for j in range(num_nodes):
                vect = corr_[j]
                top = np.flip(np.argsort(vect)[-num_neighbors:])
                mask_j = np.zeros(num_nodes)
                for t in top:
                    mask_j[t] = vect[t]
                key.append(mask_j)
            key = np.stack(key, axis=0)
            return key        
        keys = np.stack([get_key(corr[..., i]) for i in range(corr.shape[-1])], axis=-1)
    return keys
     


def get_values(keys, seq):
    # seq shape (batch_size, in_feats, num_nodes, input_horizon)
    # keys tensor of shape (num_nodes, num_nodes, num_keys)
    # output shape (batch_size, in_feats, num_nodes, num_neighbors, num_keys)
    out = []
    if isinstance(seq, torch.Tensor): 
        for k in range(keys.shape[-1]):
            out_key = []
            v = keys[..., k]
            num_nodes = v.shape[0]
            for i in range(num_nodes):
                neigh_i = torch.where(v[i]!=0)
                out_i = seq[:, :, neigh_i[0], k]
                out_key.append(out_i)
            out_key = torch.stack(out_key, dim=-2)
            out.append(out_key)
        out = torch.stack(out, dim=-1)
    else: # numpy array
        for k in range(keys.shape[-1]):
            out_key = []
            v = keys[..., k]
            num_nodes = v.shape[0]
            for i in range(num_nodes):
                neigh_i = np.where(v[i]!=0)
                out_i = seq[:, :, neigh_i[0], k]
                out_key.append(out_i)
            out_key = np.stack(out_key, axis=-2)
            out.append(out_key)
        out = np.stack(out, axis=-1)
    
    return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_q, d_Q, d_k, values_equal_keys):
        super(MultiHeadAttention, self).__init__()
        
        self.d_q = d_q
        self.d_k = d_k
        self.d_Q = d_Q
        self.values_equal_keys = values_equal_keys
        self.num_heads = num_heads
        self.K = nn.ModuleList()
        
        for _ in range(self.num_heads):
            self.K.append(
                nn.Sequential(
                nn.Linear(self.d_k, self.d_Q),
                nn.ReLU()
                )
            )       
        
        
    def forward(self, query, keys, V):
        # query shape (batch_size, in_feats, num_nodes, input_horizon)
        # keys shape (num_nodes, num_nodes, num_keys)
        # V shape (batch_size, in_feats, num_nodes, num_neighbors, num_keys)       
        
        Q = query.reshape(query.shape[0], -1)
        att_heads = []
        for h in range(self.num_heads):
            if self.values_equal_keys:
                K = torch.stack([self.K[h](keys[..., k].reshape(query.shape[0], -1)) for k in range(keys.shape[-1])], dim=-1)
                att_h = torch.einsum('ab, abc -> ac', Q, K)
            else:
                K = torch.stack([self.K[h](keys[..., k].reshape(-1)) for k in range(keys.shape[-1])], dim=-1)
                att_h = torch.einsum('ab, bc -> ac', Q, K)
            att_heads.append(att_h)
        att = torch.mean(torch.stack(att_heads, dim=0), dim=0)
        att = F.softmax(att/np.sqrt(self.d_k), dim=-1)
        
        out = torch.einsum('ae, abcde -> abcd', att, V)
        
        return out  # shape (batch_size, in_feats, num_nodes, num_neighbors)

def aggregate_correlations(corr, learn_type):
    # corr shape (num_nodes, num_nodes, num_keys)
    if learn_type == 'dot_product':
        result = corr[..., 0]
        for i in range(1, corr.shape[-1]):
            result *= corr[..., i]
    else: # sum
        result = corr[..., 0]
        for i in range(1, corr.shape[-1]):
            result += corr[..., i]
    return result

class NFE(nn.Module):
    def __init__(self, num_heads, num_neighbors, num_nodes, input_horizon, in_feats, temporal_horizons, num_spatial, keys):
        super(NFE, self).__init__()
        
        self.keys = keys
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.input_horizon = input_horizon
        self.in_feats = in_feats
        self.temporal_horizons = temporal_horizons
        self.num_spatial = num_spatial

        d_q = self.in_feats*self.num_nodes*self.input_horizon       
        d_k = num_nodes*num_nodes
        # we assume Wq and Wv as identity matrices. we do not learn them
        self.multiHeadAttention = MultiHeadAttention(num_heads, d_q, d_q, d_k, self.values_equal_keys)
      
        self.correlations = get_correlations()
        
    def forward(self, input, V): 
        # query shape (batch_size, in_feats, num_nodes, input_horizon)
        keys = self.keys # by default. in case of learning it will be overwritten       
        out = self.multiHeadAttention(input, keys, V)
        
        return out    
        

class trainer():
    def __init__(self, scaler, num_nodes, in_dim):
        
        kernel_size = int((args.input_horizon + args.num_neighbors-1)/12) + 2

        self.model = gwnet(num_nodes, in_dim, kernel_size=kernel_size)
        self.model.to(device)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = 5
            
        correlations = get_correlations()
        corrs = [np.concatenate([correlations[..., [i]], correlations[..., -num_spatial:]], axis=-1) for i in range(args.temporal_horizons)]
        learned_corr = np.stack([aggregate_correlations(corrs[i], args.corr_learn_type) for i in range(args.temporal_horizons)], axis=-1)
        keys = get_keys(learned_corr, args.num_neighbors) if args.num_neighbors < correlations.shape[1] else learned_corr 
        keys = torch.Tensor(keys).to(device)
        
        self.nfe = NFE(args.num_heads, args.num_neighbors, num_nodes, args.input_horizon, in_feats=in_dim, temporal_horizons=args.temporal_horizons,
                           num_spatial=num_spatial, keys=keys).to(device)
        params = [
                    {'params': self.model.parameters()},
                    {'params': self.nfe.parameters()}
                ]

        self.optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


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
        self.optimizer.zero_grad()
        
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
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        
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


class DataLoader(object):
    def __init__(self, xs, ys, vs, batch_size, pad_with_last_sample=True): 
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            
            v_padding = np.repeat(vs[-1:], num_padding, axis=0)
            vs = np.concatenate([vs, v_padding], axis=0)
            
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.vs = vs

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        
        vs = self.vs[permutation]
        self.vs = vs

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                v_i = self.vs[start_ind: end_ind, ...]
                yield (x_i, y_i, v_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def get_metrics(dataloader, engine, scaler, phase, ep=None):
    epoch = {"val": f"epoch_{ep}/", "test": ""}
    outputs = []
    realy = torch.Tensor(dataloader[f'y_{phase}']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for _, (x, _, v) in enumerate(dataloader[f'{phase}_loader'].get_iterator()):
        # phasex = torch.Tensor(x).to(device)  # phasex to say valx or testx
        phasex = torch.Tensor(x[..., :-1]).to(device)
        phasex = phasex.transpose(1, 3)
        
        phasev = torch.Tensor(v[..., :-1, :]).to(device)
        phasev = phasev.transpose(1, 3)        
        with torch.no_grad():
            
            phasex = engine.enrich_(phasex, phasev)
            
            phasex = nn.functional.pad(phasex, (1, 0, 0, 0))
            preds, _ = engine.model(phasex)
            preds = preds.transpose(1, 3)           
        outputs.append(preds.squeeze())
        
        del phasex, phasev, preds

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(output_horizon):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        if phase == "test":
            log = '{} - dataset: {}. Best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(args.model_name, dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        
    return amae, amape, armse
    

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def masked_mse(preds, labels, null_val=np.nan, params=None):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, params=None):
    return torch.sqrt(masked_mse(preds, labels, null_val=null_val, params=params))


def masked_mae(preds, labels, null_val=np.nan, params=None):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def load_dataset(dataset_name='metr-la', input_horizon=12, output_horizon=12, train_fraction=.7, test_fraction=.2):
    print("Getting the data ...")
    data_ = {}
    
    # get data into pivoted df of shape (datetimes, num_nodes)
    dataset_path = 'path to the dataset W (T_max X N) in .h5 format, index is the date_time'
    data = pd.read_hdf(dataset_path)

    print(df.shape)
    num_nodes = df.shape[1]
    df = df.sort_index(axis=1).reset_index()
    index = df.columns[0]
    df[index] = pd.to_datetime(df[index])
    df = df[(df[index].dt.hour >= start_hour)
            & (df[index].dt.hour <= end_hour)]
    
    # constrcuction of data split with speed and avg_speed
    x_offsets = np.arange(-input_horizon + 1, 1, 1)
    y_offsets = np.arange(1, output_horizon + 1, 1)
    num_samples = df.shape[0]
    data = [df]
    
    data = np.stack(data, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)  # x: (num_samples, input_horizon, num_nodes, input_features)
    y = np.stack(y, axis=0)  # y: (num_samples, output_horizon, num_nodes, output_features)    
    
    num_samples, input_horizon, num_nodes, input_features = x.shape

    correlations = get_correlations()
    corrs = [np.concatenate([correlations[..., [i]], correlations[..., -num_spatial:]], axis=-1) for i in range(args.temporal_horizons)]
    learned_corr = np.stack([aggregate_correlations(corrs[i], args.corr_learn_type) for i in range(args.temporal_horizons)], axis=-1)
    keys = get_keys(learned_corr, args.num_neighbors) if args.num_neighbors < num_nodes else learned_corr       
    values = get_values(keys, x.transpose(0, 3, 2, 1)).transpose(0, 3, 2, 1, 4) # shape (num_samples, num_neighbors, num_nodes, in_feats, num_keys)

    num_samples = x.shape[0]
    num_test = round(num_samples * test_fraction)
    num_train = round(num_samples * train_fraction)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    v_train, v_val, v_test = values[:num_train], values[num_train: num_train + num_val], values[-num_test:]

    # mkdir(dir_path)
    
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _v = locals()["v_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape, "v:", _v.shape)
        data_['x_' + cat], data_['y_' + cat] = _x, _y
        data_['v_' + cat] = _v
    scaler = StandardScaler(mean=data_['x_train'][..., 0].mean(), std=data_['x_train'][..., 0].std())
    
    # data format
    for category in ['train', 'val', 'test']:
        data_['x_' + category][..., 0] = scaler.transform(data_['x_' + category][..., 0])
    data_['train_loader'] = DataLoader(data_['x_train'], data_['y_train'], data_['v_train'], batch_size)
    data_['val_loader'] = DataLoader(data_['x_val'], data_['y_val'], data_['v_val'], batch_size)
    data_['test_loader'] = DataLoader(data_['x_test'], data_['y_test'], data_['v_test'], batch_size)
    data_['scaler'] = scaler
    
    return data_


class Configuration:
    def __init__(self):
        self._parser = configargparse.ArgumentParser()
        
        # config parsed by the default parser
        self._config = None

        # individual configurations for different runs
        self._configs = []
        
        # arguments with more than one value
        self._multivalue_args = []
               
        
    def parse(self):
        self._config = self._parser.parse_args()

        # find values with more than one entry
        dict_config = vars(self._config)
        for k in dict_config :
            if isinstance(dict_config[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._config)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_config[ma]:
                    current = copy.deepcopy(c)
                    setattr(current, ma, v)
                    new_configs.append(current)

            # store splitted values
            self._configs = new_configs

    def get_configs(self):
        return self._configs


def setup_config(config):
    print('Configuration setup ...')
    
    config._parser.add("-sn", "--snorm", default=1, type=int, help="spatial normalisation", nargs='*')
    config._parser.add("-tn", "--tnorm", default=1, type=int, help="temporal normalistation", nargs='*')
    config._parser.add("-oH", "--output_horizon", default=12, type=int, help="number of steps to predict", nargs='*')
    config._parser.add("-iH", "--input_horizon", default=12, type=int, help="number of historical steps", nargs='*')
    config._parser.add("-tnF", "--train_fraction", default=.7, type=float, help="fraction training samples", nargs='*')
    config._parser.add("-dN", "--dataset_name", default='metr-la', help="Name of the dataset", nargs='*') # 'pems-bay', 'hannover', 'braunschweig', 'wolfsburg'
    config._parser.add("-mod", "--model_name", default='SCANNER_plus', help="Name of the method", nargs='*')
    config._parser.add("-dv", "--device", default='cuda', help="Name of the method", nargs='*')
    config._parser.add("-ttF", "--test_fraction", default=.2, type=float, help="fraction test samples", nargs='*') 
    config._parser.add("-ep", "--epochs", default=50, type=int, help="number of epochs", nargs='*')
    config._parser.add("-sd", "--seed", default=42, type=int, help="random seed for all modules", nargs='*')
    config._parser.add("-exp", "--expid", default=-1, type=int, help="experiment id", nargs='*')
    config._parser.add("-bS", "--batch_size", default=64, type=int, help="batch size for train, val and test", nargs='*')

    config._parser.add("-nN", "--num_neighbors", default=2, type=int, help="number of neighnors to consider for each node", nargs='*')
    config._parser.add("-tH", "--temporal_horizons", default=12, type=int, help="number of past horizons of each neighbor", nargs='*')
    config._parser.add("-stH", "--start_hour", default=5, type=int, help="starting hour to filter the dataset", nargs='*')
    config._parser.add("-edH", "--end_hour", default=23, type=int, help="ending hour to filter the dataset", nargs='*')
    config._parser.add('-pt', '--patience', type=int, default=15, help='patience for early stop', nargs='*')
    config._parser.add('-nHd', '--num_heads', type=int, default=1, help='number of heads for attention', nargs='*')
    
    config.parse()


def compute(args):
    # define run specific parameters based on user input

    save_dir = 'directory in which the pretrained model will be saved'
    dataloader = load_dataset(dataset_name, args.input_horizon, output_horizon, train_fraction, test_fraction)
    scaler = dataloader['scaler']

    print('Configuration done properly ...')

    # Get Shape of Data
    _, _, num_nodes, in_feats = dataloader['x_train'].shape
    
    in_dim = in_feats - 1

    engine = trainer(scaler=scaler, num_nodes=num_nodes, in_dim=in_dim)

    print('saving directory at ' + save_dir)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    curr_min = np.inf
    wait = 0
    for i in range(1, epochs + 1):
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
            # dataloader['train_loader'].shuffle(seed=42)

            t_ = time.time()
            for iter, (x, y, v) in enumerate(dataloader['train_loader'].get_iterator()):
                # trainx = torch.Tensor(x).to(device)
                trainx = torch.Tensor(x[..., :-1]).to(device)   	# speed and time_of_day
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                
                trainv = torch.Tensor(v[..., :-1,:]).to(device)   	# speed and time_of_day
                trainv = trainv.transpose(1, 3)
                
                del x, y, v

                metrics = engine.train(trainx, trainy[:, 0, :, :], trainv)  # 0: speed
                
                del trainx, trainy, trainv

                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % print_every == 0:
                    log = '{} - dataset: {}. Iter: {:03d}, Train Loss: {:.4f}, Time: {:.2f} secs'
                    print(log.format(model_name, dataset_name, iter, train_loss[-1], (time.time()-t_)),
                        flush=True)
                    t_ = time.time()

            t2 = time.time()
            train_time.append(t2 - t1)

            mtrain_loss = np.mean(train_loss)


            # validation
            phase = "val"

            s1 = time.time()
            valid_loss, valid_mape, valid_rmse = get_metrics(dataloader=dataloader, engine=engine, scaler=scaler, phase=phase, ep=i)
            s2 = time.time()
            log = '{} - dataset: {}. Epoch: {:03d}/{:03d}, Inference Time: {:.4f} secs'
            print(log.format(model_name, dataset_name, i, epochs,(s2 - s1)))
            val_time.append(s2 - s1)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)

            his_loss.append(mvalid_loss)

            log = '{} - dataset: {}. Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}, Training Time: {:.1f} secs/epoch. Best epoch: {}, best val loss: {:.4f}'
            print(log.format(model_name, dataset_name, i, mtrain_loss, mvalid_loss, mvalid_mape,
                            mvalid_rmse, (t2 - t1), np.argmin(his_loss), np.min(his_loss)), flush=True)
            
            # save only if less than the minimum
            if mvalid_loss == np.min(his_loss):
                wait = 0
                print(f'{model_name} - dataset: {dataset_name}. Epoch: {i:03d} - Val_loss decreases from {curr_min:.4f} to {mvalid_loss:.4f}')
                curr_min = mvalid_loss
                torch.save(engine.model.state_dict(), f'{save_dir}/_id{expid}_num{exp_num}_best_.pth')
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

    # testing
    phase = "test"
    tt1 = time.time()
    engine.model.load_state_dict(torch.load(f'{save_dir}/_id{expid}_num{exp_num}_best_.pth'))

    amae, amape, armse = get_metrics(dataloader=dataloader,engine=engine, scaler=scaler, phase=phase)

    del dataloader, engine
    
    mean_amae = np.mean(amae)
    mean_amape = np.mean(amape)
    mean_armse = np.mean(armse)
    log = '{} - dataset: {}. On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(model_name, dataset_name, output_horizon, mean_amae, mean_amape, mean_armse))
      
    print('Completed')    
    


if __name__ == "__main__":
    config = Configuration()
    setup_config(config)

    exp_num = 1
    # create one thread for each c
    tot_exp = len(config.get_configs())
    print('Number of experiments: ', tot_exp)
    for args in config.get_configs():
        set_seed(args.seed)
        args.use_neighbors = 1
        
        args.expid = expid if args.expid < 0 else args.expid
                    
        print(f'Starting experiment number {exp_num}/{tot_exp} ...')
        
 
        model_name = args.model_name
        dataset_name = args.dataset_name
        batch_size = args.batch_size
        output_horizon = args.output_horizon
        weight_decay = 0.0001
        dropout = 0.3
        learning_rate = 0.001
        nhid = 32
        epochs = args.epochs
        print_every = 100
        expid = args.expid
        train_fraction = args.train_fraction
        test_fraction = args.test_fraction
        snorm_bool = args.snorm
        tnorm_bool = args.tnorm
        start_hour = args.start_hour 
        end_hour = args.end_hour 
        device = args.device

        device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        num_spatial = 1               

        print(args)   
        compute(args)
        
        exp_num += 1
