from __future__ import absolute_import, division, print_function

import math


import torch.nn as nn
from .stochastic_inits import init_stochastic_linear
from .stochastic_layers import StochasticLinear

'''   Xavier initialization
Like in PyTorch's default initializer'''

def init_layers(model, log_var_init=None):

    for m in model.modules():
        init_module(m, log_var_init)


def init_module(m, log_var_init):
    # Linear standard
    if isinstance(m, nn.Linear):
        n = m.weight.size(1)
        stdv = 1. / math.sqrt(n)
        #m.weight.data.uniform_(-stdv, stdv)
        nn.init.uniform_(m.weight.data, -1 * stdv, stdv)
        if m.bias is not None:
            #m.bias.data.uniform_(-stdv, +stdv)
            nn.init.uniform_(m.bias.data, -1 * stdv, stdv)

    # BatchNorm2d
    elif isinstance(m, nn.BatchNorm2d):
        #m.weight.data.fill_(1)
        #m.bias.data.zero_()
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)

    # Linear stochastic
    elif isinstance(m, StochasticLinear):
        init_stochastic_linear(m, log_var_init)


