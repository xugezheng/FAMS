import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.calibration import calibration_curve
import logging
from scipy import interpolate
from data.hypers import CALI_PARAMS

logger = logging.getLogger("fair")

# ----------------------------------------------------------------------------------------------
# Post Processing and Calculation
# ----------------------------------------------------------------------------------------------
# acc
def compute_accuracy(y, hat_y, bar=0.5, if_logger=False):
    """
    Computing the accuracy score of the predictor
    :param y:
    :param hat_y:
    :return:
    """
    if if_logger:
        logger.info(
            "groud truth y mean is: {}, predict y mean is: {}".format(
                y.mean(), hat_y.mean()
            )
        )
        logger.info("predict y is: {}".format(hat_y))

    y = np.where(y > bar, 1, 0)
    hat_y = np.where(hat_y > bar, 1, 0)

    if if_logger:
        logger.info(
            "The acc_bar is: {}, dataset size is: {}, groud truth y = 1 num is: {}, predict y = 1 num is: {}".format(
                bar, len(y), y.sum(), hat_y.sum()
            )
        )

    return accuracy_score(y, hat_y)


# suf gap
def new_calibration_curve(y, y_hat, params, normalize=False):

    
    y_hat_normalized = (
        (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min()) if normalize else y_hat
    )
    bins = np.linspace(0.0, 1.0 + 1e-8, params["n_bins"] + 1)
    binids = np.digitize(y_hat_normalized, bins) - 1
    # print(binids)

    bin_sums = np.bincount(binids, weights=y_hat_normalized, minlength=len(bins)-1)
    bin_true = np.bincount(binids, weights=y, minlength=len(bins)-1)
    bin_total = np.bincount(binids, minlength=len(bins)-1)

    nonzero = bin_total != 0
    bin_total[~nonzero] = 1
    prob_true = bin_true / bin_total
    prob_pred = bin_sums / bin_total
    if (
        len(prob_pred) != params["n_bins"]
        or len(prob_true) != params["n_bins"]
        or len(bin_total) != params["n_bins"]
    ):
        print(prob_true, prob_pred, bin_total)
    if params["interpolate_kind"]:
        prob_true, prob_pred = calibration_curve(y, y_hat, normalize=normalize, n_bins=params["n_bins"])

    return prob_true, prob_pred


# def standard_suf_gap_all(y_hat, y, A, prm, if_logger=False):
#     num_A = len(np.unique(A))
#     params = prm.params
#     n_bins = params["n_bins"]
#     interpolate_kind = params["interpolate_kind"]
    
#     groups_pred_true = np.zeros((num_A, n_bins))
#     groups_pred_prob = np.zeros((num_A, n_bins))
#     # all_prob_true, all_prob_pred = new_calibration_curve(y, y_hat, params)
#     all_prob_true, all_prob_pred = calibration_curve(y, y_hat, n_bins=n_bins)
#     for i in range(len(np.unique(A))):
#         try:
#             t, p = new_calibration_curve(
#                 y[A == np.unique(A)[i]], y_hat[A == np.unique(A)[i]], params
#             )
#             if params["interpolate_kind"]:
#                 # new_x = np.linspace(0.01, 0.99, n_bins)
#                 new_x = all_prob_pred
#                 f = interpolate.interp1d(
#                             p,
#                             t,
#                             bounds_error=False,
#                             fill_value=(t[0], t[-1]),
#                             kind=interpolate_kind,
#                         )
#                 t = f(new_x)
#                 p = new_x
#             groups_pred_true[i] = t
#             groups_pred_prob[i] = p
#         except:
#             continue
            
#     exp_x_given_a = np.abs(all_prob_true - groups_pred_true).mean(axis=1)
#     if 'toxic' == prm.dataset:
#         exp_x_a = exp_x_given_a[1:].mean()
#         if if_logger:
#             logger.info("[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
#         return exp_x_a
#     else:
#         exp_x_a = exp_x_given_a.mean()
#         if if_logger:
#             logger.info("[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
#         # logger.info("[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
#         return exp_x_a

def standard_suf_gap_all(y_hat, y, A, prm, if_logger=False):
    num_A = len(np.unique(A))
    params = prm.params
    n_bins = params["n_bins"]
    interpolate_kind = params["interpolate_kind"]

    groups_pred_true = np.zeros((num_A, n_bins))
    groups_pred_prob = np.zeros((num_A, n_bins))
    all_prob_true, all_prob_pred = calibration_curve(y, y_hat, n_bins=n_bins)
    if all_prob_true.shape[0] != n_bins:
        all_prob_true = np.zeros(n_bins)
    for i in range(len(np.unique(A))):
        try:
            t, p = calibration_curve(
                y[A == np.unique(A)[i]], y_hat[A ==
                                               np.unique(A)[i]], n_bins=n_bins
            )
            if params["interpolate_kind"]:
                # new_x = np.linspace(0.01, 0.99, n_bins)
                new_x = all_prob_pred
                f = interpolate.interp1d(
                    p,
                    t,
                    bounds_error=False,
                    fill_value=(t[0], t[-1]),
                    kind=interpolate_kind,
                )
                t = f(new_x)
                p = new_x
            groups_pred_true[i] = t
            groups_pred_prob[i] = p
        except:
            continue

    exp_x_given_a = np.abs(all_prob_true - groups_pred_true).mean(axis=1)
    if 'toxic' in prm.dataset.lower():
        exp_x_a = exp_x_given_a[1:].mean()
        if if_logger:
            logger.info(
                "[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
        return exp_x_a
    elif 'adult' in prm.dataset.lower():
        exp_x_a = exp_x_given_a[1:].mean()
        if if_logger:
            logger.info(
                "[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
        return exp_x_a
    else:
        exp_x_a = exp_x_given_a.mean()
        if if_logger:
            logger.info(
                "[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
        logger.info(
            "[SufGAP] The average sufficiency gap : {}".format(exp_x_a))
        return exp_x_a


# all
def result_show(y_test, predict, A_test, prm):
    logger.info("=============== Accuracy =============")
    accuracy = compute_accuracy(y_test, predict, prm.acc_bin)
    logger.info("[Accuracy] The overall accuracy is: {}".format(accuracy))


    logger.info("=============== Sufficient Gap =============")
    suf_gap_avg_score = standard_suf_gap_all(predict, y_test, A_test, prm, if_logger=True)
    logger.info("[SufGAP] The overall sufficiency gap is: {}".format(suf_gap_avg_score))

    return (
        accuracy,
        suf_gap_avg_score,
    )


def result_wandb(y_test, predict, A_test, prm):
    accuracy = compute_accuracy(y_test, predict, prm.acc_bin, if_logger=False)
    suf_gap_avg_score = standard_suf_gap_all(predict, y_test, A_test, prm, if_logger=False)

    wandb_dict = {
        "acc": accuracy,
        "suf_gap_avg": suf_gap_avg_score,
    }
    return wandb_dict
