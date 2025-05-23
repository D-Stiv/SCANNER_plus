import numpy as np
import pickle
import torch



def get_correlations(args):
    # transform distances in correlations and put diagonal to zero.
    
    corr = []

    if args.temporal_horizons > 0:
        # temporal correlation
        with open('[Path to the temporal correlation matrices B]', "rb") as f:
            data = pickle.load(f)
        _, _, temp_corr = data
        temp_corr = abs(temp_corr)
        temp_horizon = np.min([args.temporal_horizons, temp_corr.shape[0]])
        for i in range(temp_horizon):
            c_i = temp_corr[i]
            corr.append(c_i)
    
    if args.distance:
        # spatial distance
        with open('[Path to the spatial distance matrices A]', "rb") as f:
            data = pickle.load(f)
        _, _, dist = data
        
        # min-max normalizeation inversing (0 dist --> 1 corelation)
        dist_corr = (np.max(dist) - dist)/(np.max(dist) - np.min(dist))
        corr.append(dist_corr)
        
    corr = np.stack(corr, axis=-1)
    return corr  


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
                mask_j = torch.zeros(num_nodes).to(corr.device)
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
    # seq offset = [-H_I+1, -H_I+2, ..., 0] --> long term are lower indices and short term are higher indices
    out = []
    if isinstance(seq, torch.Tensor): 
        for k in range(keys.shape[-1]-1, -1, -1):
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
        for k in range(keys.shape[-1]-1, -1, -1):
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
