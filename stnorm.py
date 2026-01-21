# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.


import torch
import torch.nn as nn


class TemporalNorm(nn.Module):
    """
    Temporal Normalization (TN)
    Normalizes over the temporal dimension (T)
    """

    def __init__(self, num_nodes, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        # Learnable affine parameters (per node, per feature)
        self.gamma = nn.Parameter(torch.ones(1, num_nodes, 1, num_features))
        self.beta  = nn.Parameter(torch.zeros(1, num_nodes, 1, num_features))

    def forward(self, x):
        """
        x: (B, N, T, C)
        """
        # Mean & variance over time
        mean = x.mean(dim=2, keepdim=True)        # (B, N, 1, C)
        var  = x.var(dim=2, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return x_norm * self.gamma + self.beta
    

class SpatialNorm(nn.Module):
    """
    Spatial Normalization (SN)
    Normalizes over the spatial dimension (N)
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        # Shared affine parameters (per feature)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, num_features))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, num_features))

    def forward(self, x):
        """
        x: (B, N, T, C)
        """
        # Mean & variance over nodes
        mean = x.mean(dim=1, keepdim=True)        # (B, 1, T, C)
        var  = x.var(dim=1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return x_norm * self.gamma + self.beta


class STNorm(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.tn = TemporalNorm(num_nodes, num_features)
        self.sn = SpatialNorm(num_features)

    def forward(self, x):
        """
        x: (B, N, T, C)
        """
        x_tn = self.tn(x)
        x_sn = self.sn(x)

        # Concatenate along feature dimension
        return torch.cat([x_tn, x_sn], dim=-1)
