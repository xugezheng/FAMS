import logging
import numpy as np
import datetime
import importlib
import time
import os.path as osp
import copy

__all__ = ["TxtLogger"]


def TxtLogger(filename, verbosity="info", logname="fair"):
    level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        # "critical": logging.CRITICAL,
    }
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(logname)
    logger.setLevel(level_dict[verbosity])
    # file handler
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def set_logger(prm):
    now = int(round(time.time() * 1000))
    log_file = f"{prm.method}_{prm.dataset}_seed_{prm.seed}_{str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000)))}.log"
    log_dir = osp.abspath(osp.join(
        osp.dirname(__file__),
        "../logs",
        prm.dataset,
        prm.exp_name,
    ))
    return log_dir, log_file


def set_wandb(prm):
    pre_config = copy.deepcopy(vars(prm))
    del (
        pre_config["use_wandb"],
        pre_config["train_inf_step"],
        pre_config["params"],
    )


    pre_config["interpolate_kind"] = prm.params["interpolate_kind"]
    pre_config["sufgapcali"] = prm.params["n_bins"]

    wandb_name = f"{prm.method}_seed_{prm.seed}_sensattr_{prm.sens_attrs}_task_{prm.N_subtask}_lr_{prm.lr_prior}_{prm.lr_post}_epoch_{prm.training_epoch}_batch_{prm.batch_size}_klweight_{prm.weight}_{prm.exp_name}"
    project_name = f"fair_metabayes_{prm.exp_name}_{prm.dataset}"
    
    return pre_config, project_name, wandb_name


def set_npy(prm, cur_epoch):
    npy_dir = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../npy",
            prm.dataset,
            prm.exp_name,
            prm.sens_attrs,
            f"{prm.seed}_{cur_epoch}",
        )
    )
    npy_file_pre = f"{prm.method}_{prm.dataset}_seed_{prm.seed}_task_{prm.N_subtask}_epoch_{cur_epoch}_batch_{prm.batch_size}_{prm.exp_name}"

    return npy_dir, npy_file_pre
