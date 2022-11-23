from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .stochastic_inits import init_stochastic_linear
from utils.common import list_mult

# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
# -------------------------------------------------------------------------------------------
class StochasticLayer(nn.Module):
    # base class of stochastic layers with re-parametrization
    # self.init  and self.operation should be filled by derived classes

    def create_stochastic_layer(self, weights_shape, bias_size):
        # create the layer parameters
        # values initialization is done later
        self.weights_shape = weights_shape
        self.weights_count = list_mult(weights_shape)
        if bias_size is not None:
            self.weights_count = self.weights_count + bias_size
        self.w_mu = get_param(weights_shape)
        self.w_log_var = get_param(weights_shape)
        self.w = {'mean': self.w_mu, 'log_var': self.w_log_var}
        if bias_size is not None:
            self.b_mu = get_param(bias_size)
            self.b_log_var = get_param(bias_size)
            self.b = {'mean': self.b_mu, 'log_var': self.b_log_var}


    def forward(self, x):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)
        # self.operation should be linear or conv

        if self.use_bias:
            b_var = torch.exp(self.b_log_var)
            bias_mean = self.b['mean']
        else:
            b_var = None
            bias_mean = None

        out_mean = self.operation(x, self.w['mean'], bias=bias_mean)

        eps_std = self.eps_std
        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            out_var = self.operation(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            #noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            noise = eps_std * torch.randn_like(out_mean, requires_grad=False)

            # out_var = F.relu(out_var) # to avoid nan due to numerical errors
            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out

    def set_eps_std(self, eps_std):
        old_eps_std = self.eps_std
        self.eps_std = eps_std
        return old_eps_std

# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
# -------------------------------------------------------------------------------------------
class StochasticLinear(StochasticLayer):


    def __init__(self, in_dim, out_dim, prm, use_bias=True):
        super(StochasticLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        self.use_bias = use_bias
        if use_bias:
            bias_size = out_dim
        else:
            bias_size = None
        self.create_stochastic_layer(weights_size, bias_size)
        init_stochastic_linear(self, prm.log_var_init)
        self.eps_std = prm.eps_std

    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def operation(self, x, weight, bias):
        out = F.linear(x, weight, bias)
        return out

# -------------------------------------------------------------------------------------------
#  Auxilary functions
# -------------------------------------------------------------------------------------------
def get_param(shape):
    # create a parameter
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.empty(*shape))