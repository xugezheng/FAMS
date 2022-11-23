
from __future__ import absolute_import, division, print_function


import torch
import math
from layers.stochastic_layers import StochasticLayer


# -----------------------------------------------------------------------------------------------------------#


# compute the KL divergence between the prior-model and post-model, we will use model for multiple times
def get_KLD(prior_model, post_model, prm, noised_prior=False):

    prior_layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]
    post_layers_list = [layer for layer in post_model.children() if isinstance(layer, StochasticLayer)]

    total_dvrg = 0
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]
        if hasattr(prior_layer, 'w'):
            total_dvrg += get_dvrg_element(post_layer.w, prior_layer.w, prm, noised_prior)
        if hasattr(prior_layer, 'b'):
            total_dvrg += get_dvrg_element(post_layer.b, prior_layer.b, prm, noised_prior)

    #if hasattr(prm, 'divergence_type') and prm.divergence_type == 'W_NoSqr':
    #    total_dvrg = torch.sqrt(total_dvrg)

    return total_dvrg
# -------------------------------------------------------------------------------------------


def  get_dvrg_element(post, prior, prm, noised_prior=False):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""

    if noised_prior and prm.kappa_post > 0:
        prior_log_var = add_noise(prior['log_var'], prm.kappa_post)
        prior_mean = add_noise(prior['mean'], prm.kappa_post)
    else:
        prior_log_var = prior['log_var']
        prior_mean = prior['mean']

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior_log_var)
    post_std = torch.exp(0.5 * post['log_var'])
    prior_std = torch.exp(0.5 * prior_log_var)

    if not hasattr(prm, 'divergence_type') or prm.divergence_type == 'KL':
        numerator = (post['mean'] - prior_mean).pow(2) + post_var
        denominator = prior_var
        div_elem = 0.5 * torch.sum(prior_log_var - post['log_var'] + numerator / denominator - 1)

    elif prm.divergence_type in ['W_Sqr', 'W_NoSqr']:
        # Wasserstein norm with p=2
        # according to DOWSON & LANDAU 1982
        div_elem = torch.sum((post['mean'] - prior_mean).pow(2) + (post_std - prior_std).pow(2))
    else:
        raise ValueError('Invalid prm.divergence_type')

    # note: don't add small number to denominator, since we need to have zero KL when post==prior.
    assert div_elem >= 0
    return div_elem
# -------------------------------------------------------------------------------------------

def add_noise(param, std):

    return param + torch.tensor(param.data.new(param.size()).normal_(0, std), requires_grad=False)
# -------------------------------------------------------------------------------------------






