# MIT License
#
# Original work:
# Copyright (c) <2019> <Zonghan Wu>
#
# Modifications:
# Copyright (c) 2026 D-Stiv
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ----
# Modifications points: 
# - integreted STNorm in gwnet: lines 62; 81; 152
# - inserted correlation and neighborhood feature enrichment: lines 174-288

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from stnorm import STNorm
from correlations import get_correlations, get_keys, get_values


class gwnet(nn.Module):
    def __init__(self, device, args, num_nodes, adjinit, in_dim, supports, output_horizon, nhid, dropout, adaptadj, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        out_dim = output_horizon
        residual_channels = nhid
        dilation_channels = nhid
        skip_channels = nhid * 8
        end_channels = nhid * 16
        aptinit = adjinit
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.adaptadj = args.adaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.stnorm = nn.ModuleList()
        num = 3

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.stnorm.append(STNorm(num_nodes, residual_channels))

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

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            x_list = []
            x_list.append(x)
            
            # STNorm
            x_list.append(self.stnorm[i](x))
            
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
        att = F.softmax(att/np.sqrt(self.d_k), dim=-1)  # (num_samples, input_horizon)
        
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
    def __init__(self, args, num_heads, num_neighbors, num_nodes, input_horizon, in_feats, temporal_horizons, num_spatial, corr_learn_type, filter_agg_type, values_equal_keys, not_learned_keys=None):
        super(NFE, self).__init__()
        
        self.not_learned_keys = not_learned_keys
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.input_horizon = input_horizon
        self.in_feats = in_feats
        self.temporal_horizons = temporal_horizons
        self.num_spatial = num_spatial
        self.corr_learn_type = corr_learn_type
        self.filter_agg_type = filter_agg_type
        self.values_equal_keys = values_equal_keys
        
        if self.corr_learn_type == 'linear_layer':
            self.correlation_learning = nn.ModuleList()
            for _ in range(self.temporal_horizons):
                self.correlation_learning.append(
                    nn.Sequential(
                        nn.Linear(self.num_spatial+1, 1),
                        nn.Sigmoid()
                    )
                )

        
        if self.filter_agg_type == 'attention_block':
            d_q = self.in_feats*self.num_nodes*self.input_horizon
            if self.values_equal_keys:  
                d_k = self.in_feats*self.num_nodes*self.num_neighbors
            else:         
                d_k = num_nodes*num_nodes
            # we assume Wq and Wv as identity matrices. we do not learn them
            self.multiHeadAttention = MultiHeadAttention(num_heads, d_q, d_q, d_k, self.values_equal_keys)
        else:
            self.linear_layer = nn.Sequential(nn.Linear(self.temporal_horizons, 1), nn.ELU()) # ELU instead of ReLU because normalize speed could be negative but not too much
        
        self.correlations = get_correlations(args)
        
    def forward(self, input, V): 
        # query shape (batch_size, in_feats, num_nodes, input_horizon)
        keys = self.not_learned_keys # by default. in case of learning it will be overwritten
        corrs = [torch.cat(([torch.Tensor(self.correlations[..., [i]]).to(input.device), torch.Tensor(self.correlations[..., -self.num_spatial:]).to(input.device)]), dim=-1) for i in range(self.temporal_horizons)]
        if self.corr_learn_type == 'linear_layer':
            # learning the correlation
            learned_corr = torch.cat([self.correlation_learning[i](corrs[i]) for i in range(self.temporal_horizons)], dim=-1)       
            keys = get_keys(learned_corr, self.num_neighbors) if self.num_neighbors < self.num_nodes else learned_corr
            values = get_values(keys, input) # shape (batch_size, in_feats, num_nodes, num_neighbors, num_keys)
            V = values        
        
        if self.filter_agg_type == 'attention_block':
            if self.values_equal_keys:
                out = self.multiHeadAttention(input, V, V)
            else:
                out = self.multiHeadAttention(input, keys, V)
        else:
            out = self.linear_layer(V)
            out = torch.squeeze(out, dim=-1)
        
        return out    
        