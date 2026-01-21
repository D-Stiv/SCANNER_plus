# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
