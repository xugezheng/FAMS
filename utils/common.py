from __future__ import absolute_import, division, print_function

from datetime import datetime
import os
import torch.nn as nn
import torch
import numpy as np
import random
import sys
import pickle
from functools import reduce

# -----------------------------------------------------------------------------------------------------------#
# General auxilary functions
# -----------------------------------------------------------------------------------------------------------#
def list_mult(L):
    return reduce(lambda x, y: x*y, L)

# -----------------------------------------------------------------------------------------------------------#
def seed_setup(seed, deep_fix=False, block_cudnn=False):
   
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deep_fix:
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False 
        if block_cudnn:
            torch.backends.cudnn.enabled = False 

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



