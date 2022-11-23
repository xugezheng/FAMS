# the code is inspired by: https://github.com/katerakelly/pytorch-maml

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import list_mult
from .stochastic_layers import StochasticLinear, StochasticLayer
from .layer_inits import init_layers

# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------
def count_weights(model):
    # note: don't counts batch-norm parameters
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            count += list_mult(m.weight.shape)
            if hasattr(m, 'bias'):
                count = count + list_mult(m.bias.shape)
        elif isinstance(m, StochasticLayer):
            count += m.weights_count
    return count


#  -------------------------------------------------------------------------------------------
#  Main function
#  -------------------------------------------------------------------------------------------
def get_model(prm, model_type='Stochastic'):

    model_name = prm.model_name

    # Define default layers functions
    def linear_layer(in_dim, out_dim, use_bias=True):
        if model_type == 'Standard':
            return nn.Linear(in_dim, out_dim, use_bias)
        elif model_type == 'Stochastic':
            return StochasticLinear(in_dim, out_dim, prm, use_bias)

    #  Return selected model:
    if model_name == 'FcNet3':
        model = FcNet3(model_type, model_name, linear_layer, prm)
    elif model_name == 'FcNet4':
        model = FcNet4(model_type, model_name, linear_layer, prm)
    else:
        raise ValueError('Invalid model_name')

    # Move model to device (GPU\CPU):
    model.to(prm.device)

    # init model:
    init_layers(model, prm.log_var_init)

    model.weights_count = count_weights(model)

    return model

#  -------------------------------------------------------------------------------------------
#   Base class for all stochastic models
# -------------------------------------------------------------------------------------------
class general_model(nn.Module):
    def __init__(self):
        super(general_model, self).__init__()

    def set_eps_std(self, eps_std):
        old_eps_std = None
        for m in self.modules():
            if isinstance(m, StochasticLayer):
                old_eps_std = m.set_eps_std(eps_std)
        return old_eps_std

    def _init_weights(self, log_var_init):
        init_layers(self, log_var_init)




# -------------------------------------------------------------------------------------------
# Models collection
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
#  3-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet3(general_model):
    def __init__(self, model_type, model_name, linear_layer, prm):
        super(FcNet3, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.layers_names = ('FC1', 'FC2', 'FC_out')
        input_size = prm.input_shape
        output_dim = prm.output_dim


        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 100
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc_out = linear_layer(n_hidden2, output_dim)

        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x.squeeze(dim=-1))
        return x

# -------------------------------------------------------------------------------------------
#  4-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet4(general_model):
    def __init__(self, model_type, model_name, linear_layer, prm):
        super(FcNet4, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.layers_names = ('FC1', 'FC2', 'FC3', 'FC_out')
        input_size = prm.input_shape
        output_dim = prm.output_dim


        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 200
        n_hidden3 = 100
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc3 = linear_layer(n_hidden2, n_hidden3)
        self.fc_out = linear_layer(n_hidden3, output_dim)

        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x.squeeze(dim=-1))
        return x


