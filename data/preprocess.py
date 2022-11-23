import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
import wilds
import h5py
import pandas as pd


def get_distilbert_embedding(input_list, path=None):
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertModel,
        DistilBertTokenizerFast,
    )

    model_kwargs = {}
    model_name = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path:

        pretrained_model_path = osp.abspath(path)
        model_name = "distilbert-base-uncased"

        raw_statedict = torch.load(pretrained_model_path, map_location=device)
        for key in list(raw_statedict["algorithm"]):
            if key.startswith("model."):
                new_key = key[6:]
                raw_statedict["algorithm"][new_key] = raw_statedict["algorithm"].pop(
                    key
                )
        model_kwargs["state_dict"] = raw_statedict["algorithm"]

    model = DistilBertModel.from_pretrained(model_name, **model_kwargs).to(device)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    print(model.config)
    print("model loaded")

    embedding = []
    for str in tqdm(input_list):
        inputs = tokenizer(str, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embedding.append(outputs["last_hidden_state"][0, 0, :].detach().cpu().numpy())
    return np.stack(embedding)


# ============================================ AMAZON==================================================
def amazon_data_info(train_path, test_path):
    train_h5 = h5py.File(osp.abspath(train_path))
    test_h5 = h5py.File(osp.abspath(test_path))
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
    train_users, train_each_textnum = np.unique(A_train, return_counts=True)
    test_users, test_each_textnum = np.unique(A_test, return_counts=True)
    print(
        f"Train Info:\ntrain users number is {len(train_users)}, listed: {train_users}, each text number is {train_each_textnum}, minimum is {train_each_textnum.min()}\ntest users number is {len(test_users)}, listed are {test_users}, each text num is {test_each_textnum}."
    )

    train_y, train_each_y = np.unique(Y_train, return_counts=True)
    test_y, test_each_y = np.unique(Y_test, return_counts=True)
    print(
        f"Train Info:\ntrain users number is {len(train_y)}, listed: {train_y}, each text number is {train_each_y}\ntest users number is {len(test_y)}, listed are {test_y}, each text num is {test_each_y}."
    )


def get_distilbert_embedding_amazon(root_dir, path=None):
    full_dataset = wilds.get_dataset(dataset="amazon", root_dir=root_dir)
    train_set = full_dataset.get_subset(split="train")
    test_set = full_dataset.get_subset(split="id_test")
    train = [
        (train_set[i][0], train_set[i][1], train_set[i][2][0])
        for i in range(len(train_set))
    ]
    test = [
        (test_set[i][0], test_set[i][1], test_set[i][2][0])
        for i in range(len(test_set))
    ]
    train_text, train_Y, train_A = zip(*train)
    test_text, test_Y, test_A = zip(*test)

    if path is None:
        train_out = "amazon_wofinetune_train.h5"
        test_out = "amazon_wofinetune_test.h5"
    else:
        train_out = "amazon_train.h5"
        test_out = "amazon_test.h5"

    out_amazon_train_h5 = osp.abspath(osp.join(root_dir, train_out))
    out_amazon_test_h5 = osp.abspath(osp.join(root_dir, test_out))

    train_X = get_distilbert_embedding(list(train_text), path)
    with h5py.File(out_amazon_train_h5, "w") as f:
        f.create_dataset("X", data=train_X)
        f.create_dataset("user", data=np.array(train_A, dtype=np.int))
        f.create_dataset("Y", data=np.array(train_Y, dtype=np.int))

    test_X = get_distilbert_embedding(list(test_text), path)
    with h5py.File(out_amazon_test_h5, "w") as f:
        f.create_dataset("X", data=test_X)
        f.create_dataset("user", data=np.array(test_A, dtype=np.int))
        f.create_dataset("Y", data=np.array(test_Y, dtype=np.int))


def get_amazon_data_info(root_dir, path=None):
    full_dataset = wilds.get_dataset(dataset="amazon", root_dir=root_dir)
    train_set = full_dataset.get_subset(split="train")
    val_set = full_dataset.get_subset(split="val")
    test_set = full_dataset.get_subset(split="id_test")
    train = [
        (train_set[i][0], train_set[i][1], train_set[i][2][0])
        for i in range(len(train_set))
    ]
    val = [
        (val_set[i][0], val_set[i][1], val_set[i][2][0]) for i in range(len(val_set))
    ]
    test = [
        (test_set[i][0], test_set[i][1], test_set[i][2][0])
        for i in range(len(test_set))
    ]
    train_text, train_Y, train_A = zip(*train)
    val_text, val_Y, val_A = zip(*val)
    test_text, test_Y, test_A = zip(*test)

    train_users, train_each_textnum = np.unique(train_A, return_counts=True)
    val_users, val_each_textnum = np.unique(val_A, return_counts=True)
    test_users, test_each_textnum = np.unique(test_A, return_counts=True)
    print(
        f"Train Info:\ntrain users number is {len(train_users)}, avg text num is {train_each_textnum.mean()}, max text number is {train_each_textnum.max()}, minimum is {train_each_textnum.min()}."
    )
    print(
        f"Val Info:\nval users number is {len(val_users)}, avg text num is {val_each_textnum.mean()}, max text number is {val_each_textnum.max()}, minimum is {val_each_textnum.min()}."
    )
    print(
        f"Test Info:\ntest users number is {len(test_users)}, avg text num is {test_each_textnum.mean()}, max text number is {test_each_textnum.max()}, minimum is {test_each_textnum.min()}."
    )

    train_y, train_each_y = np.unique(train_Y, return_counts=True)
    # val_y, val_each_y = np.unique(val_Y, return_counts=True)
    test_y, test_each_y = np.unique(test_Y, return_counts=True)
    print(
        f"Train Info:\ntrain users number is {len(train_y)}, train y distribution is {train_each_y}\ntest users number is {len(test_y)}, test y distribution is {test_each_y}."
    )


# ======================================== TOXIC =============================================
def get_distilbert_embedding_from_wildstoxic(input, out_dir, path=None):
    out_file_name = osp.abspath(osp.join(out_dir, "toxic_from_wilds_all.h5"))
    df = pd.read_csv(input)
    X = get_distilbert_embedding(list(df["comment_text"]), path)
    Y = df["toxicity"].to_numpy()
    with h5py.File(out_file_name, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)


if __name__ == "__main__":

    data_dir = osp.abspath(osp.join(osp.dirname(__file__), "../DATASOURCE"))
    ckpt_dir = osp.abspath(osp.join(osp.dirname(__file__), "../checkpoints"))

    # # amazon
    # root_dir = osp.join(data_dir, 'amazon')
    # get_distilbert_embedding_amazon(root_dir)
    # get_amazon_data_info(root_dir, path=None)

    # change to your own model path
    # train_path = osp.join(data_dir, 'amazon', 'amazon_train.h5')
    # test_path = osp.join(data_dir, 'amazon', 'amazon_test.h5')
    # amazon_data_info(train_path, test_path)

    # change to your own model path
    # # toxic
    # input = osp.join(data_dir, 'toxic', 'all_data_with_identities.csv')
    # out_dir = osp.join(data_dir, 'toxic')
    # path = osp.join(ckpt_dir, 'civilcomments_seed_0_epoch_best_model.pth')
    # get_distilbert_embedding_from_wildstoxic(input, out_dir, path)
