import torch
import torch.nn as nn

# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random
import math
from embedder import NLP_embedder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import time
from data import load_data, SimpleDataset
from torch.utils.data import DataLoader
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'

    

def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)

    max_epochs = 10

    def evaluate(classifier, X,Y):
        y_pred = classifier.predict(X)
        if not (type(y_pred) == np.ndarray):
           y_pred = np.argmax(y_pred.to("cpu"), axis=1)
        accuracy = accuracy_score(Y, y_pred)
        evaluation_dict = {}
        evaluation_dict["accuracy"] = accuracy
        return evaluation_dict
    
    def train_fn():
        @ag.args(
                 hidden_fc=ag.space.Int(lower=5, upper=200),
                 number_layers = ag.space.Int(lower = 1, upper = 10),
                 lr=ag.space.Real(lower=5e-7, upper=1e-4, log=True),
                 CNNs = ag.space.Categorical(ag.space.Dict(hidden_fc=ag.space.Int(lower=5, upper=200),
                                                           number_layers = ag.space.Int(lower = 1, upper = 5),
                                                           kernel_size = ag.space.Int(lower = 3, upper = 11)
                                                        #   pooling = ag.space.Categorical("max", "mean", "first")
                     ),ag.space.Dict()
                 
                 ),
                 pooling = ag.space.Categorical("max", "mean", "[CLS]"),
              #   layer_norm = ag.space.Categorical("layer_norm", "Batch_norm", "none"),
                 freeze_base = ag.space.Categorical(True,False),
                 Attention = ag.space.Categorical(ag.space.Dict(#embed_dim=ag.space.Categorical(8,16,32,64,128,256,512),
                                                                num_heads=ag.space.Categorical(1,2,4,8,16),
                                                           number_layers = ag.space.Int(lower = 1, upper = 5)
                     ),ag.space.Dict())
                )
        def run_opaque_box(args, reporter):
            print(args)
            model = NLP_embedder(num_classes = 2,batch_size = 32,args =  args)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name="sst2")
            
            for e in range(max_epochs):
                model.fit(X_train, Y_train, epochs=1)
            
                evaluation_dict = evaluate(model, X_val, Y_val)
                 
                reporter(objective=evaluation_dict[ACTIVE_METRIC_NAME], epoch=e)
            
        return run_opaque_box

    runboxfn = train_fn()
    
    
    


    # Create scheduler and searcher:
    # First get_config are random, the remaining ones use constrained BO
    search_options = {
        'random_seed': SEED,
        'num_fantasy_samples': 5,
        'num_init_random': 1,
        'debug_log': True}

    myscheduler = ag.scheduler.HyperbandScheduler(   
        runboxfn,
        resource={'num_cpus': 4, 'num_gpus': 1},
        searcher='bayesopt',
        search_options=search_options,
        num_trials=50,
        reward_attr=REWARD_ATTR_NAME,
        time_attr='epoch',
        grace_period=1,
        reduction_factor=3,
        max_t=15,
        brackets=1,
        checkpoint=config["DEFAULT"]["directory"] + "/checkpoint.ckp"
        # constraint_attr=CONSTRAINT_METRIC_NAME
    )

    # Run HPO experiment
    print("run scheduler")
    myscheduler.run()
    myscheduler.join_jobs()
    
    print("best config", myscheduler.get_best_config())
    print("best reward", myscheduler.get_best_reward())
    print("best task id", myscheduler.get_best_task_id())
    
    
    myscheduler.get_training_curves(filename=config["DEFAULT"]["directory"]+"/training_curves.png")
    
    
    
    
    search_metric == "custom"
    myscheduler.run_with_config(myscheduler.get_best_config())
    




