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
import time
from data import load_data, SimpleDataset
from torch.utils.data import DataLoader
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = ["cola", "sst2", "mrpc", "mnli"]#"qqp", "rte" 
    

def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)

    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])
        # dataset specific
    dataset = config["DEFAULT"]["dataset"]
    baseline = config["DEFAULT"]["baseline"] == "True"
    combined = config["DEFAULT"]["combined"] == "True"
    print("dataset:", dataset)
    print("trying all datasets at once: ", combined)
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3
        
    if baseline :
        class dummy():
            def __init__(self):
                return
        args = dummy()
        args.hidden_fc = 100
        args.number_layers = 1
        args.lr = 2e-5
        args.pooling = "[CLS]"
        args.CNNs = {}
        args.Attention = {}
        model = NLP_embedder(num_classes = num_classes,batch_size = batch_size,args =  args)
        print("build model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print("load dataset")
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
        
        print("fitting baseline model")
        accuracy = model.fit(X_train, Y_train, epochs=max_epochs, X_val = X_val, Y_val = Y_val)
        
        print("accuracy", accuracy)
    torch.cuda.empty_cache()
    
    def train_fn():
        @ag.args(
                 hidden_fc=ag.space.Int(lower=5, upper=200),
                 number_layers = ag.space.Int(lower = 1, upper = 5),
                 lr=ag.space.Real(lower=1e-5, upper=1e-4, log=True),
                 CNNs = ag.space.Categorical(ag.space.Dict(hidden_fc=ag.space.Int(lower=5, upper=200),
                                                           number_layers = ag.space.Int(lower = 1, upper = 5),
                                                           kernel_size = ag.space.Categorical(3,5,7,9,11),
                                                           skip = ag.space.Categorical(True,False)
                                                        #   pooling = ag.space.Categorical("max", "mean", "first")
                     ),ag.space.Dict()
                 
                 ),
#                 LSTMs = ag.space.Categorical(ag.space.Dict(hidden_fc=ag.space.Int(lower=5, upper=200),
#                                                            number_layers=ag.space.Int(lower=1, upper=5)
#                                                 
#                     ),ag.space.Dict()
#                   
#                 ),
                 
                 pooling = ag.space.Categorical("max", "mean", "[CLS]"),
              #   layer_norm = ag.space.Categorical("layer_norm", "Batch_norm", "none"),
               #  freeze_base = ag.space.Categorical(True,False),
                 Attention = ag.space.Categorical(ag.space.Dict(#embed_dim=ag.space.Categorical(8,16,32,64,128,256,512),
                                                                num_heads=ag.space.Categorical(1,2,4,8,16),
                                                           number_layers = ag.space.Int(lower = 1, upper = 5)
                     ),ag.space.Dict())
                )
        def run_opaque_box(args, reporter):
          #  args.freeze_base = False 
            print(args)
            print(combined)
            if combined:
                cum_accuracy = 0.0
                for datasetnew in datasets:
                    if "small" in dataset:
                        datasetnew = datasetnew + "small"
                    num_classes = 2
                    if "mnli" in datasetnew:
                        num_classes = 3
                    model = NLP_embedder(num_classes = num_classes,batch_size = batch_size,args =  args)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    
                    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=datasetnew)
                    
                    acc  = model.fit(X_train, Y_train, epochs=max_epochs, X_val = X_val, Y_val = Y_val, reporter = None)
                    cum_accuracy += acc
                torch.cuda.empty_cache()
                print("cumulative accuracy", cum_accuracy)
                reporter(objective=cum_accuracy)
            else:
                num_classes = 2
                if "mnli" in dataset:
                    num_classes = 3
                print("loading model")
                model = NLP_embedder(num_classes = num_classes,batch_size = batch_size,args =  args)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                print("loading dataset")
                X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
                print("training model")
                model.fit(X_train, Y_train, epochs=max_epochs, X_val = X_val, Y_val = Y_val, reporter = reporter)
                
            
        return run_opaque_box

    runboxfn = train_fn()
    
    
    


    # Create scheduler and searcher:
    # First get_config are random, the remaining ones use constrained BO
    search_options = {
        'random_seed': SEED,
        'num_fantasy_samples': 5,
        'num_init_random': 1,
        'debug_log': True}

    if combined:
        myscheduler = ag.scheduler.FIFOScheduler(   
            runboxfn,
            resource={'num_cpus': 4, 'num_gpus': 1},
            searcher='bayesopt',
            search_options=search_options,
            num_trials=int(config["DEFAULT"]["num_trials"]),
            reward_attr=REWARD_ATTR_NAME,
            checkpoint=config["DEFAULT"]["directory"] + "/checkpoint.ckp"
            # constraint_attr=CONSTRAINT_METRIC_NAME
        )
    else:
        myscheduler = ag.scheduler.HyperbandScheduler(   
            runboxfn,
            resource={'num_cpus': 4, 'num_gpus': 1},
            searcher='bayesopt',
            search_options=search_options,
            num_trials=int(config["DEFAULT"]["num_trials"]),
            reward_attr=REWARD_ATTR_NAME,
            time_attr='epoch',
            grace_period=1,
            reduction_factor=3,
            max_t=15,
            brackets=1,
            checkpoint=config["DEFAULT"]["directory"] + "/checkpoint.ckp"
            # constraint_attr=CONSTRAINT_METRIC_NAME
        )


#     init_config = {'Attention▁0▁num_heads▁choice': 0, 
#                    'Attention▁0▁number_layers': 3, 
#                    'Attention▁choice': 1,
# #                     'LSTMs▁0▁hidden_fc' : 100,
# #                     'LSTMs▁0▁number_layers': 1, 
# #                     'LSTMs▁choice': 0, 
#                     'CNNs▁0▁hidden_fc': 102, 
#                     'CNNs▁0▁kernel_size': 7, 
#                     'CNNs▁0▁number_layers': 3, 
#                     'CNNs▁0▁skip▁choice': 0, 
#                     'CNNs▁choice': 1, 
#                     'freeze_base▁choice': 1, 
#                     'hidden_fc': 102,
#                      'lr': 2e-5, 
#                      'number_layers': 1, 
#                      'pooling▁choice': 2}
#     myscheduler.run_with_config(init_config)
    
  #  Run HPO experiment
    print("run scheduler")
    myscheduler.run()
    myscheduler.join_jobs()
     
    print("best config", myscheduler.get_best_config())
    print("best reward", myscheduler.get_best_reward())
    print("best task id", myscheduler.get_best_task_id())
     
     
    myscheduler.get_training_curves(filename=config["DEFAULT"]["directory"]+"/training_curves.png")
    
    
   # myscheduler.run_with_config(myscheduler.get_best_config())
    




