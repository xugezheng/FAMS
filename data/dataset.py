import pdb
import os.path as osp
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import h5py
import logging

from data.hypers import EXTRACTED_ATTRS_MAPPING

logger = logging.getLogger("fair")


def _quantization_binning(data, num_bins=10):
    qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
    bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
    bin_widths = np.diff(bin_edges, axis=0)
    bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
    return bin_edges, bin_centers, bin_widths


def _quantize(inputs, bin_edges, num_bins=10):
    quant_inputs = np.zeros(inputs.shape[0])
    for i, x in enumerate(inputs):
        quant_inputs[i] = np.digitize(x, bin_edges)
    quant_inputs = quant_inputs.clip(1, num_bins) - 1  # Clip edges
    return quant_inputs


def _one_hot(a, num_bins=10):
    return np.squeeze(np.eye(num_bins)[a.reshape(-1).astype(np.int32)])


def DataQuantize(X, bin_edges=None, num_bins=10):
    """
    Quantize: First 4 entries are continuos, and the rest are binary
    """
    X_ = []
    for i in range(5):

        # Xi_q_max = max(X[:,i])
        # Xi_q_min = min(X[:,i])
        # Xi_q = X[:,i] - Xi_q_min/ (Xi_q_max-Xi_q_min)
        # Xi_q = X[:,i]

        if bin_edges is not None:
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        else:
            bin_edges, bin_centers, bin_widths = _quantization_binning(
                X[:, i], num_bins
            )
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        Xi_q = _one_hot(Xi_q, num_bins)
        X_.append(Xi_q)

    for i in range(5, len(X[0])):
        if i == 39:  # gender attribute
            continue
        Xi_q = _one_hot(X[:, i], num_bins=2)
        X_.append(Xi_q)

    return np.concatenate(X_, 1), bin_edges


def general_info_logging(X_train, X_test, A_train, A_test, Y_train, Y_test):
    train_users, train_each_textnum = np.unique(A_train, return_counts=True)
    test_users, test_each_textnum = np.unique(A_test, return_counts=True)
    train_y, train_each_y = np.unique(Y_train, return_counts=True)
    test_y, test_each_y = np.unique(Y_test, return_counts=True)

    logger.info("SELECTED Data Info:")

    logger.info(
        "SELECTED Train Data: {} example with dim {}, y mean is {}".format(
            X_train.shape[0], X_train.shape[1], Y_train.mean()
        )
    )
    logger.info(
        f"SELECTED TRAIN [USER] are {len(train_users)}, maximum text per user is {train_each_textnum.max()}, minimum text per user is {train_each_textnum.min()}, average text per user is {train_each_textnum.mean()}."
    )
    logger.info(
        f"SELECTED TRAIN [LABEL] are {train_y}, distribution is {train_each_y}."
    )

    logger.info(
        "SELECTED Test Data: {} example with dim {}, y mean is {}".format(
            X_test.shape[0], X_test.shape[1], Y_test.mean()
        )
    )
    logger.info(
        f"SELECTED TEST [USER] are {len(test_users)}, maximum text per user is {test_each_textnum.max()}, minimum text per user is {test_each_textnum.min()}, average text per user is {test_each_textnum.mean()}."
    )
    logger.info(
        f"SELECTED TEST [LABEL] are {test_y}, distribution is {test_each_y}.")

    logger.info(f"Added Information:")
    logger.info(f"Train Data:")
    logger.info(f"E(Y) = {Y_train.mean()}")
    logger.info(
        f"E(Y|a) = {np.array([Y_train[A_train==i].mean() for i in train_users]).mean()}"
    )
    logger.info(
        f"| E(Y|a)_train - E(Y|a)_train | mean = {np.array([np.abs(Y_train[A_train==i].mean() - Y_train.mean()) for i in train_users]).mean()}"
    )
    logger.info(f"Test Data:")
    logger.info(f"E(Y) = {Y_test.mean()}")
    logger.info(
        f"E(Y|a) = {np.array([Y_test[A_test==i].mean() for i in test_users]).mean()}"
    )
    logger.info(
        f"| E(Y|a)_test - E(Y|a)_test | mean = {np.array([np.abs(Y_test[A_test==i].mean() - Y_test.mean()) for i in test_users]).mean()}"
    )
    return


# ----------------------------------------------------------------------------------------------
# Get Data
# ----------------------------------------------------------------------------------------------
def load_data(prm):
    logger.info("DATA LOADING...")
    logger.info("DATASET: {}".format(prm.dataset))
    if prm.dataset == "toxic":
        return load_toxic_distilbert(prm.sens_attrs, prm.acc_bar)
    elif prm.dataset == "amazon":
        return load_amazon_distilbert(
            prm.N_subtask, prm.lower_rate, prm.upper_rate
        )
    elif prm.dataset == "adult":
        return preprocess_adult_data(seed=prm.seed)
    elif prm.dataset == "celeba":
        return load_celeba_features()
    else:
        raise ValueError("Invalid dataset:{}".format(prm.dataset))


# ----------------------------------------------------------------------------------------------
# Adult Dataset
# ----------------------------------------------------------------------------------------------
def get_adult_data():
    """
    We borrow the code from https://github.com/IBM/sensitive-subspace-robustness
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
    """

    headers = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-stataus",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "y",
    ]
    data_dir = osp.abspath(osp.dirname(__file__))
    train = pd.read_csv(
        osp.abspath(osp.join(data_dir, "../DATASOURCE/adult/adult.data")), header=None
    )
    test = pd.read_csv(
        osp.abspath(osp.join(data_dir, "../DATASOURCE/adult/adult.test")), header=None
    )
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df["y"] = df["y"].replace(
        {" <=50K.": 0, " >50K.": 1, " >50K": 1, " <=50K": 0})

    df = df.drop(df[(df[headers[-2]] == " ?") |
                 (df[headers[6]] == " ?")].index)
    df = pd.get_dummies(
        df,
        columns=[
            headers[1],
            headers[5],
            headers[6],
            headers[7],
            headers[9],
            headers[8],
            "native-country",
        ],
    )

    delete_these = [
        "race_ Amer-Indian-Eskimo",
        "race_ Asian-Pac-Islander",
        "race_ Black",
        "race_ Other",
        "sex_ Female",
    ]

    delete_these += [
        "native-country_ Cambodia",
        "native-country_ Canada",
        "native-country_ China",
        "native-country_ Columbia",
        "native-country_ Cuba",
        "native-country_ Dominican-Republic",
        "native-country_ Ecuador",
        "native-country_ El-Salvador",
        "native-country_ England",
        "native-country_ France",
        "native-country_ Germany",
        "native-country_ Greece",
        "native-country_ Guatemala",
        "native-country_ Haiti",
        "native-country_ Holand-Netherlands",
        "native-country_ Honduras",
        "native-country_ Hong",
        "native-country_ Hungary",
        "native-country_ India",
        "native-country_ Iran",
        "native-country_ Ireland",
        "native-country_ Italy",
        "native-country_ Jamaica",
        "native-country_ Japan",
        "native-country_ Laos",
        "native-country_ Mexico",
        "native-country_ Nicaragua",
        "native-country_ Outlying-US(Guam-USVI-etc)",
        "native-country_ Peru",
        "native-country_ Philippines",
        "native-country_ Poland",
        "native-country_ Portugal",
        "native-country_ Puerto-Rico",
        "native-country_ Scotland",
        "native-country_ South",
        "native-country_ Taiwan",
        "native-country_ Thailand",
        "native-country_ Trinadad&Tobago",
        "native-country_ United-States",
        "native-country_ Vietnam",
        "native-country_ Yugoslavia",
    ]

    delete_these += ["fnlwgt", "education"]

    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(
        df=df, label_names=["y"], protected_attribute_names=["sex_ Male", "race_ White"]
    )


def preprocess_adult_data(seed=0):
    """
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)
    Input: seed: the seed used to split data into train/test
    """
    # Get the dataset and split into train and test
    dataset_orig = get_adult_data()

    # we will standardize continous features
    continous_features = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    continous_features_indices = [
        dataset_orig.feature_names.index(feat) for feat in continous_features
    ]

    # get a 80%/20% train/test split
    dataset_orig_train, dataset_orig_test = dataset_orig.split(
        [0.8], shuffle=True, seed=seed
    )
    SS = StandardScaler().fit(
        dataset_orig_train.features[:, continous_features_indices]
    )
    dataset_orig_train.features[:, continous_features_indices] = SS.transform(
        dataset_orig_train.features[:, continous_features_indices]
    )
    dataset_orig_test.features[:, continous_features_indices] = SS.transform(
        dataset_orig_test.features[:, continous_features_indices]
    )

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    X_val = X_train[: len(X_test)]
    y_val = y_train[: len(X_test)]
    X_train = X_train[len(X_test):]
    y_train = y_train[len(X_test):]

    # gender id = 39
    A_train = X_train[:, 39]
    A_val = X_val[:, 39]
    A_test = X_test[:, 39]

    X_train, bin_edges = DataQuantize(X_train)
    X_val, _ = DataQuantize(X_val, bin_edges)
    X_test, _ = DataQuantize(X_test, bin_edges)

    return X_train, X_test, A_train, A_test, np.squeeze(y_train), np.squeeze(y_test)


# ----------------------------------------------------------------------------------------------
# Toxic Dataset
# ----------------------------------------------------------------------------------------------


def generate_multi_attrs_toxic(h5_file, csv_file, extracted_attrs, bar):

    all_df = pd.read_csv(csv_file)
    all_h5 = h5py.File(h5_file)

    # print(type(train_inx))
    logger.info("TOXIC dataset Preprocessing ...")
    sum_col_num = np.zeros(len(all_df))
    for group in extracted_attrs:
        logger.info(f"## {group}")
        group_col_bool = np.full(len(all_df), False)
        for attr in extracted_attrs[group]:
            attr_bool = all_df[attr] > 0.5
            group_col_bool = group_col_bool | attr_bool
        logger.info(
            f"{group} examples num is: {group_col_bool.astype(int).sum()}")
        all_df[f"{group}_added_label"] = group_col_bool
        sum_col_num += group_col_bool.astype(int)
    sum_col_bool = np.where(sum_col_num == 1, True, False)
    all_df["new_groups"] = np.zeros(len(all_df))
    out_dict = {0: "others"}
    for i, group in enumerate(extracted_attrs):
        cur_group_bool = all_df[f"{group}_added_label"] & sum_col_bool
        all_df.loc[cur_group_bool, "new_groups"] = i + 1
        out_dict[i + 1] = group
    logger.info(f"Task Dict: {out_dict}")
    logger.info(f"Grouped Info: {all_df['new_groups'].value_counts()}")
    for s in ["train", "test", "val"]:
        logger.info(
            f"{s}: \n{all_df.loc[all_df['split'] == s, 'new_groups'].value_counts()}"
        )
    for i in out_dict:
        cur_df = all_df.loc[all_df["new_groups"] == i]
        # print(cur_df.head(5))
        logger.info(f"{out_dict[i]} toxicity is {cur_df['toxicity'].mean()}")

    train_inx = np.flatnonzero(all_df["split"] == "train")
    test_inx = np.flatnonzero(all_df["split"] == "test")
    X = np.array(all_h5["X"])
    Y = np.array(all_h5["Y"])
    X_train = X[train_inx]
    Y_train = Y[train_inx]
    A_train = all_df["new_groups"].to_numpy()[train_inx]
    X_test = X[test_inx]
    Y_test = Y[test_inx]
    A_test = all_df["new_groups"].to_numpy()[test_inx]

    Y_train = np.where(Y_train > bar, 1, 0)
    Y_test = np.where(Y_test > bar, 1, 0)
    logger.info("TOXIC dataset Loaded.")

    return X_train, X_test, A_train, A_test, Y_train, Y_test


def load_toxic_distilbert(sens_attrs, bar):

    h5_file = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../DATASOURCE/toxic/toxic_from_wilds_all.h5",
        )
    )

    csv_file = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../DATASOURCE/toxic/all_data_with_identities.csv",
        )
    )

    extracted_attrs = EXTRACTED_ATTRS_MAPPING[sens_attrs]
    logger.info(f"TOXIC dataset pre-processed.")
    logger.info(
        f"Sensitive Attribute is {sens_attrs}, group is {extracted_attrs}")
    logger.info(f"Accuracy bar is {bar}")
    X_train, X_test, A_train, A_test, Y_train, Y_test = generate_multi_attrs_toxic(
        h5_file, csv_file, extracted_attrs, bar
    )
    general_info_logging(X_train, X_test, A_train, A_test, Y_train, Y_test)

    return X_train, X_test, A_train, A_test, Y_train, Y_test


# ----------------------------------------------------------------------------------------------
# Amazon Dataset
# ----------------------------------------------------------------------------------------------
def amazon_selection(users_selected, X, A, Y, lower_rate, upper_rate):
    idx = list(
        set(np.where(np.isin(A, users_selected))[0])
        & (set(np.where(Y >= upper_rate)[0]) | set(np.where(Y <= lower_rate)[0]))
    )
    Y = Y[idx]
    Y = np.where(Y >= upper_rate, 1, 0)
    return X[idx], A[idx], Y


def load_amazon_distilbert(group_num=50, lower_rate=1, upper_rate=4):
    train_fname = "amazon_train.h5"
    test_fname = "amazon_test.h5"


    train_h5 = h5py.File(
        osp.abspath(
            osp.join(osp.dirname(__file__),
                     "../DATASOURCE/amazon", train_fname)
        )
    )
    test_h5 = h5py.File(
        osp.abspath(osp.join(osp.dirname(__file__),
                    "../DATASOURCE/amazon", test_fname))
    )

    X_train, A_train, Y_train = (
        np.array(train_h5["X"]),
        np.array(train_h5["user"]),
        np.array(train_h5["Y"]),
    )
    X_test, A_test, Y_test = (
        np.array(test_h5["X"]),
        np.array(test_h5["user"]),
        np.array(test_h5["Y"]),
    )

    # logger info
    train_users, train_each_textnum = np.unique(A_train, return_counts=True)
    test_users, test_each_textnum = np.unique(A_test, return_counts=True)
    train_y, train_each_y = np.unique(Y_train, return_counts=True)
    test_y, test_each_y = np.unique(Y_test, return_counts=True)
    logger.info("Amazon dataset Preprocessing ...")
    logger.info("Original Data Info:")
    logger.info(
        "ORIGINAL Train Data: {} example with dim {}, y mean is {}".format(
            X_train.shape[0], X_train.shape[1], Y_train.mean()
        )
    )
    logger.info(
        f"ORIGINAL TRAIN [USER] are {len(train_users)}, minimum text per user is {train_each_textnum.min()}."
    )
    logger.info(
        f"ORIGINAL TRAIN [LABEL] are {train_y}, distribution is {train_each_y}."
    )
    logger.info(
        "ORIGINAL Test Data: {} example with dim {}, y mean is {}".format(
            X_test.shape[0], X_test.shape[1], Y_test.mean()
        )
    )
    logger.info(
        f"ORIGINAL TEST [USER] are {len(test_users)}, minimum text per user is {test_each_textnum.min()}."
    )
    logger.info(
        f"ORIGINAL TEST [LABEL] are {test_y}, distribution is {test_each_y}.")

    users_selected = np.random.choice(
        np.unique(A_test), size=group_num, replace=False)
    logger.info(f"Selected [{group_num}] USER are: {users_selected}")

    X_train, A_train, Y_train = amazon_selection(
        users_selected, X_train, A_train, Y_train, lower_rate, upper_rate
    )
    X_test, A_test, Y_test = amazon_selection(
        users_selected, X_test, A_test, Y_test, lower_rate, upper_rate
    )
    logger.info("Amazon dataset loaded.")
    general_info_logging(X_train, X_test, A_train, A_test, Y_train, Y_test)

    return X_train, X_test, A_train, A_test, Y_train, Y_test


# ----------------------------------------------------------------------------------------------
# Celeba Dataset
# ----------------------------------------------------------------------------------------------
def load_celeba_features():
    train_fname = f'celeba_33_train.h5'
    test_fname = f'celeba_33_test.h5'

    train_h5 = h5py.File(
        osp.abspath(
            osp.join(osp.dirname(__file__),
                     "../DATASOURCE/celeba", train_fname)
        )
    )
    test_h5 = h5py.File(
        osp.abspath(osp.join(osp.dirname(__file__),
                    "../DATASOURCE/celeba", test_fname))
    )

    X_train, A_train, Y_train = (
        np.array(train_h5["X"]),
        np.array(train_h5["A"]),
        np.array(train_h5["Y"]),
    )
    X_test, A_test, Y_test = (
        np.array(test_h5["X"]),
        np.array(test_h5["A"]),
        np.array(test_h5["Y"]),
    )

    # logger info
    train_y, train_each_y = np.unique(Y_train, return_counts=True)
    test_y, test_each_y = np.unique(Y_test, return_counts=True)
    logger.info("Celeba dataset Preprocessing ...")
    logger.info("Original Data Info:")
    logger.info(
        "ORIGINAL Train Data: {} example with dim {}, y mean is {}".format(
            X_train.shape[0], X_train.shape[1], Y_train.mean()
        )
    )
    logger.info(
        f"ORIGINAL TRAIN [LABEL] are {train_y}, distribution is {train_each_y}."
    )
    logger.info(
        "ORIGINAL Test Data: {} example with dim {}, y mean is {}".format(
            X_test.shape[0], X_test.shape[1], Y_test.mean()
        )
    )
    logger.info(
        f"ORIGINAL TEST [LABEL] are {test_y}, distribution is {test_each_y}.")

    logger.info("Celeba dataset loaded.")
    general_info_logging(X_train, X_test, A_train, A_test, Y_train, Y_test)

    return X_train, X_test, A_train, A_test, Y_train, Y_test

# ----------------------------------------------------------------------------------------------
# Batch Generalization - Sampling
# ----------------------------------------------------------------------------------------------
def sample_batch_sen_idx(X, A, y, batch_size, s):
    batch_idx = np.random.choice(
        np.where(A == s)[0],
        size=batch_size,
        replace=(len(np.where(A == s)[0]) < batch_size),
    ).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y



if __name__ == "__main__":
    # load_amazon_distilbert()
    X_train, X_test, A_train, A_test, y_train, y_test = preprocess_adult_data(seed=0)
    print(X_train.shape)
    print(A_train[0:100])
