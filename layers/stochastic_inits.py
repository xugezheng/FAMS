

from __future__ import absolute_import, division, print_function


import math
import torch.nn as nn

# -----------------------------------------------------------------------------------------------------------#
# Inits
# -----------------------------------------------------------------------------------------------------------#
'''   Xavier initialization
Like in PyTorch's default initializer'''

def init_stochastic_linear(m, log_var_init):
    n = m.w_mu.size(1)
    stdv = math.sqrt(1. / n)
    #m.w_mu.data.uniform_(-stdv, stdv)
    nn.init.uniform_(m.w_mu.data,-1*stdv,stdv)
    if m.use_bias:
        #m.b_mu.data.uniform_(-stdv, stdv)
        nn.init.uniform_(m.b_mu.data, -1*stdv,stdv)
        nn.init.normal_(m.b_log_var.data,log_var_init['mean'], log_var_init['std'])
        #m.b_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])
    #m.w_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])
    nn.init.normal_(m.w_log_var.data, log_var_init['mean'], log_var_init['std'])

