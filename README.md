# Code Repo for *On Learning Fairness and Accuracy for Multiple Subgroups*

## Getting Started

### Repo Structure

```bash
.
│  all_train.py         # main train script
│  requirements.txt     # environment requirement
│
├─checkpoints           # PLM .pth model params. If directly using data provided in DATASOURCE, do not need any model checkpoints
│
├─DATASOURCE            # DATA folder 
│  ├─adult 
│  │    adult.data
│  │    adult.test
│  ├─amazon
│  │    amazon_train.h5
│  │    amazon_test.h5
│  ├─celeba
│  │    celeba_33_train.h5
│  │    celeba_33_test.h5
│  └─toxic
│       toxic_from_wilds_all.h5
│       all_data_with_identities.csv
│
├─data                  # module for data processing
│    dataset.py
│    hypers.py
│    preprocess.py
│
├─engine                # module for training (all methods)
│    fair_training.py
│
├─EXPS                  # traiing config
│    test_amazon_template.yml
│    test_toxic_template.yml
│
├─layers                # module for model construction
│    layer_inits.py
│    stochastic_inits.py
│    stochastic_layers.py
│    stochastic_models.py
│  
├─logs                  # LOG folder
│  ├─amazon
│  └─toxic
│
├─npy                   # NPY OUTPUT folder
│  ├─amazon
│  └─toxic
│
└─utils                 # Auxiliary tools
      Bayes_utils.py
      common.py
      complexity_terms.py
      figures.ipynb
      loggers.py
      optim.py
      postprocessing.py
      __init__.py
    
```

### Requirements - env

The algorithme is implemented mainly based on PyTorch Deep Learning Framework, and the datasets utilized are from [WILDS](https://wilds.stanford.edu/).

Please refer to [WILDS get started](https://wilds.stanford.edu/get_started/) or use the following command to prepare your env.

```bash
pip install -r requirements.txt
```

### Requirements - Data

- Datasets
  - WILDS - amazon
  - WILDS - civilcomments
  - Adult
  - Celeba

## Train

### Train from scratch

1. Preparing textual data from the WILDS package

    ```python
    from wilds import get_dataset
    dataset = get_dataset(dataset="amazon", download=True) # civilcomments
    ```

2. Preparing your representative data

    Using PLMs to fine-tune or directly extract the embedding features as representative input. (Fine-tuned DistillBert checkpoints can be found [here](https://drive.google.com/drive/folders/1WQqqFp7niY-Ny0sC-4BRIBX4GS_kng4W?usp=share_link).)

    Save them as `h5` file in `./DATASOURCE/#your_dataset`

3. Fast Train

### Fast Train

1. Use our provided data files and Make Sure that there are well-prepared representative data in the `./DATASOURCE/#your_dataset` folder.

   The DATASOURCE can be downloaded [here](https://drive.google.com/drive/folders/1q1Yfzz9Gp7cQlrvQR14WdI0NUJrCnQeR?usp=share_link).
   - For *amazon* data set, it needs the train and test `h5` data files (*amazon_train.h5, amazon_test.h5*).
   - For *civilcomments* data set, it need the test `h5` data files and the processed CSV file `all_data_with_identities.csv` from WILDS package (*toxic_from_wilds_all.h5*, *all_data_with_identities.csv*).
   - For *adult* data set, it needs two data files (*adult.data, adult.test*).
   - For *celeba* data set, it needs the train and test `h5` data files (*celeba_33_train.h5*, *celeba_33_test.h5*).


2. Use `all_train.py` script to train your model.
   1. Use config file in `./EXPS` to train your model.

        ```cmd
        python all_train.py -config EXPS/amazon_template.yml
        ```

        We have provided our template configs on *Amazon*, *CivilComments*, *Adult* and  *Celeba*  datasets.

        Here is an example for *Amazon* dataset:

        ``` yml

        # COMMON args
        method: ours
        model_name: FcNet4
        training_epoch: 80
        batch_size: 50
        lr_prior: 0.001 # main model lr
        #------------------------------------------------------------------------------------
        #DATASET related
        dataset: amazon
        sens_attrs: all 
        N_subtask: 200
        acc_bar: 0.5 # toxic
        lower_rate: 3 # amazon
        upper_rate: 4 # amazon
        #------------------------------------------------------------------------------------
        # METHOD Specific
        lr_post: 0.8
        weight: 0.4
        divergence_type: W_Sqr  # W_Sqr KL
        kappa_prior: 0.01
        kappa_post: 0.001
        log_var_init_mean: 0.01 #-0.1
        log_var_init_var: 0.01 #0.1
        eps_std: 0.08
        n_MC: 5
        #------------------------------------------------------------------------------------
        # CUSTOMER
        seed: 4
        train_inf_step: 2
        use_wandb: False
        wandb_username: YOURWANDBNAME
        exp_name: amazon_200_example
        #------------------------------------------------------------------------------------
        #POST
        # acc
        acc_bin: 0.5
        # suf gap
        params:
            n_bins: 10
            interpolate_kind: 'linear'  
        ```

   2. Directly passing the changeable parameters to the training script as:

        ```cmd
        python all_train.py --method ours --dataset amazon --N_subtask 200
        ```

    Note: The config file has higher priority than the direct passed params.

3. Check the result:

   1. Log file will be output to `./logs` folder, log info includes the dataset statistical result and model *test acc*, *calibration score* and *calibration gap*
   2. Numerical test result will be saved to `./npy` folder  

   Note: We have provided the `wandb` interface to trace the training process and visualize the experimental results. To use the `wandb`, please follow the instruction on [Weights&Biases](https://wandb.ai/site) official site and set `use_wandb` as `True` for `all_train.py` script.
