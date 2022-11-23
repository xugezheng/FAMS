from __future__ import absolute_import, division, print_function

import logging
import torch
from utils.complexity_terms import get_KLD
import numpy as np
from layers.stochastic_models import get_model
import torch.optim as optim
from data.dataset import sample_batch_sen_idx
from utils.postprocessing import compute_accuracy, result_wandb
from utils.optim import PostMultiStepLR, PriorExponentialLR
from utils.loggers import set_wandb, set_npy
import wandb
import copy
import os
import os.path as osp
import random
import time


logger = logging.getLogger("fair")


# ----------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------

# freeze and activate gradient w.r.t. parameters
def model_freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def model_activate(model):
    for param in model.parameters():
        param.requires_grad = True


# main training
def train_one_task(
    prior_model,
    post_model,
    loss_criterion,
    optimizer_post,
    batch,
    prm,
    task_index,
    epoch_id,
):
    """
    :param prior_model: the prior_model (only one-model)
    :param post_model: the posterior_model within task a
    :param criterion: the loss function
    :param optimizer_prior: optimizer for the prior model (contains a list of optimizers)
    :param task_batch: the data batch that contains t-tasks.
    :param prm: parameter configuration
    :return:
    """

    for _ in range(prm.max_inner):

        model_freeze(prior_model)
        model_activate(post_model)
        post_model.train()

        avg_empiric_loss = 0

        # number of MC sample (default as 5)
        n_MC = prm.n_MC
        inputs, targets = batch

        # Monte-Carlo loop in estimation prediction loss
        for i_MC in range(n_MC):
            # Empirical Loss on current task:
            outputs = post_model(inputs)
            avg_empiric_loss_curr = loss_criterion(outputs, targets)
            avg_empiric_loss = avg_empiric_loss + \
                (1 / n_MC) * avg_empiric_loss_curr
        # end Monte-Carlo loop

        # Compute the complexity term (noised prior can be set false)
        complexity = get_KLD(prior_model, post_model, prm, noised_prior=True)

        loss = avg_empiric_loss + prm.weight * complexity

        optimizer_post.zero_grad()
        loss.backward()
        optimizer_post.step()

        # freeze the posterior model
        model_freeze(post_model)
        # log
        if _ + 1 == prm.max_inner:
            model_num = prm.model_num if prm.method == "ours_sample" else prm.N_subtask
            logger.info(
                "Epoch=[{}/{}], Inner Loop Task=[{}/{}], Inner Step=[{}/{}], CLS_LOSS = {}, DIS_LOSS = {}, TOTAL_LOSS = {}".format(
                    epoch_id + 1,
                    prm.training_epoch,
                    task_index + 1,
                    model_num,
                    _ + 1,
                    prm.max_inner,
                    avg_empiric_loss.item(),
                    complexity.item(),
                    loss.item(),
                )
            )
            if prm.use_wandb and task_index <= 5:
                name = prm.method + "_task_" + str(task_index) + "_all_loss"
                wandb.log({name: loss.item()}, commit=False)


def update_meta_prior(list_of_post_model, prior_model, optimizer_prior, prm, epoch_id):

    for _ in range(prm.max_outer):

        model_activate(prior_model)
        complexity = 0.0
        kld_list = []

        for posterior in list_of_post_model:

            kld = get_KLD(prior_model, posterior, prm, noised_prior=True)
            complexity = complexity + kld
            kld_list.append(str(round(kld.item(), 4)))
        # back+prob with posterior

        optimizer_prior.zero_grad()
        complexity.backward()
        optimizer_prior.step()

        if _ + 1 == prm.max_outer:
            kld_str = "; ".join(kld_list)
            logger.info(
                "Epoch=[{}/{}], Outer Loop Step=[{}/{}], KLD for EACH Qa: {}".format(
                    epoch_id + 1, prm.training_epoch, _ + 1, prm.max_outer, kld_str
                )
            )

            if prm.use_wandb:
                wandb.log(
                    {"prior_model_loss": complexity.item() / prm.N_subtask})


def training_task_batches(
    list_of_post_model,
    list_of_post_optimizer,
    prior_model,
    prior_optimizer,
    loss_criterion,
    data_batch,
    prm,
    epoch_id,
    optimizer_prior_schedular,
    list_optimizer_post_schedular,
):
    """
    :param list_of_post_model:
    :param list_of_post_optimizer:
    :param prior_model:
    :param prior_optimizer:
    :param data_batch:
    :param loss_criterion:
    :param prm:
    :return:
    """

    for task_index, data in enumerate(data_batch):

        posterior = list_of_post_model[task_index]
        posterior_opt = list_of_post_optimizer[task_index]

        # training task
        train_one_task(
            prior_model,
            posterior,
            loss_criterion,
            posterior_opt,
            data,
            prm,
            task_index,
            epoch_id,
        )

        list_optimizer_post_schedular[task_index].step()

    # Then updating prior distribution
    update_meta_prior(list_of_post_model, prior_model,
                      prior_optimizer, prm, epoch_id)
    if prm.dataset in ["adult"]:
        if epoch_id >= 60:
            optimizer_prior_schedular.step()
    elif prm.dataset in ["celeba"]:
        pass
    else:
        optimizer_prior_schedular.step()

    logger.info(
        f"current prior optimizer lr={prior_optimizer.param_groups[0]['lr']}")
    logger.info(
        f"current post optimizer lr={list_of_post_optimizer[0].param_groups[0]['lr']}"
    )


def update_meta_post(list_of_post_model, prior_model, ratio):
    for i in range(len(list_of_post_model)):
        prob = random.uniform(0, 1)
        if prob < ratio:
            list_of_post_model[i] = copy.deepcopy(prior_model)


def train_ours(prm, prior_model, loss_criterion, X_train, A_train, y_train, X_test=None, A_test=None, y_test=None):
    # prior model
    optimizer_prior = optim.Adagrad(prior_model.parameters(), lr=prm.lr_prior)
    # optimizer schedular
    optimizer_prior_schedular = PriorExponentialLR(
        optimizer_prior, prm.training_epoch)

    # post model
    model_num = prm.N_subtask
    post_models = [get_model(prm) for _ in range(model_num)]
    list_optimizer_post = [
        optim.Adagrad(post_model.parameters(), lr=prm.lr_post)
        for post_model in post_models
    ]
    list_optimizer_post_schedular = [
        PostMultiStepLR(list_optimizer_post[i], prm.training_epoch)
        for i in range(model_num)
    ]

    for epoch_id in range(prm.training_epoch):

        batch = [
            sample_batch_sen_idx(
                X_train, A_train, y_train, prm.batch_size, np.unique(A_train)[
                    t_num]
            )
            for t_num in range(prm.N_subtask)
        ]

        time_s = time.time()

        training_task_batches(
                post_models,
                list_optimizer_post,
                prior_model,
                optimizer_prior,
                loss_criterion,
                batch,
                prm,
                epoch_id,
                optimizer_prior_schedular,
                list_optimizer_post_schedular,
            )


        time_e = time.time()

        if epoch_id % prm.train_inf_step == 0:
            ss_time = time_e - time_s
            logger.info(
                "The training time for one epoch is: {} seconds".format(
                    ss_time)
            )
            predict = inference(prior_model, X_test, prm)
            accuracy = compute_accuracy(y_test, predict, 0.5)
            logger.info(
                "The overall accuracy of EPOCH [{}] is: {}".format(
                    epoch_id, accuracy)
            )
            if prm.use_wandb:
                wandb_dict = result_wandb(y_test, predict, A_test, prm)
                wandb.log(wandb_dict, commit=False)

            if epoch_id == int(prm.training_epoch / 2):
                npy_dir, npy_file_pre = set_npy(prm, epoch_id)

                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)

                np.save(osp.join(npy_dir, npy_file_pre + "_testy.npy"), y_test)
                np.save(osp.join(npy_dir, npy_file_pre + "_testA.npy"), A_test)
                np.save(osp.join(npy_dir, npy_file_pre + "_predict.npy"), predict)


def train(
    prm,
    prior_model,
    loss_criterion,
    X_train,
    A_train,
    y_train,
    X_test=None,
    A_test=None,
    y_test=None,
):

    logger.info("===============Training Process=============")

    # wandb setting
    if prm.use_wandb:

        pre_config, project_name, wandb_name = set_wandb(prm)

        wandb.init(
            project=project_name,
            entity=prm.wandb_username,
            config=pre_config,
            reinit=True,
            name=wandb_name,
        )

    # initial test
    predict = inference(prior_model, X_test, prm)
    accuracy = compute_accuracy(y_test, predict, 0.5)
    logger.info("The Initial overall accuracy is: {}".format(accuracy))
    if prm.use_wandb:
        wandb_dict = result_wandb(y_test, predict, A_test, prm)
        wandb.log(wandb_dict)

    # wandb.finish()

    # training switch
    
    train_ours(prm, prior_model, loss_criterion, X_train,
                A_train, y_train, X_test, A_test, y_test)
    

    if prm.use_wandb:
        wandb.finish()


# ----------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------


def inference(model, inputs, prm, n_Mc=5):
    """
    :param stochastic_model:
    :param inputs:
    :param i_Mc: iteration of monte-carlo (default = 5)
    :return:
    """
    n_Mc = prm.n_MC
    inputs = torch.tensor(inputs).cuda().float()
    model_freeze(model)
    model.eval()
    output_final = 0


    for i_MC in range(n_Mc):
        outputs = model(inputs)
        output_final += (1 / n_Mc) * outputs


    model_activate(model)
    model.train()
    result = output_final.data.cpu().numpy()

    return result

