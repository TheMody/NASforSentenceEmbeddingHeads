import torch
import torch.nn as nn

# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random
import math
from embedder import NLP_embedder, hybrid_classifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from robustness import robustness_nlp_model, robustness_noise, robustness_data_shift
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import time
from data import load_data, SimpleDataset
from torch.utils.data import DataLoader
from custom_utils import Prediction_timed, time_sigmoid, time_score, score_scaled
from plot import plotpareto
from fairness import evaluate_model_bias
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'obj_metric'
CONSTRAINT_METRIC_NAME = 'constr_metric'
REWARD_ATTR_NAME = 'objective'


    

def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)


    supported_models = ['bert-base-uncased', 'google/electra-small-discriminator', 'distilbert-base-uncased', 'gpt2']#, 'bag_of_words']
    supported_classifiers = ['nn', 'svm']
    models = []
    classifiers = []

    # dataset specific
    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3


    accuracy_wf = float(config["DEFAULT"]["accuracy_wf"])
    fairness_wf = float(config["DEFAULT"]["fairness_wf"])
    robustness_wf = float(config["DEFAULT"]["robustness_wf"])
    time_wf = float(config["DEFAULT"]["time_wf"])
    wished_time = float(config["DEFAULT"]["wished_time"])
    wished_accuracy = float(config["DEFAULT"]["wished_accuracy"])
    wished_robustness = float(config["DEFAULT"]["wished_robustness"])
    Big_C = float(config["DEFAULT"]["Big_C"])
    batch_size = int(config["DEFAULT"]["batch_size"])
    max_epochs = int(config["DEFAULT"]["epochs"])
    search_metric = config["DEFAULT"]["search_metric"]
    print("batch size: ", batch_size)
    print("epochs: ", max_epochs)
    

    num_cpu = int(config["DEFAULT"]["num_cpu"])
    num_gpu = int(config["DEFAULT"]["num_gpu"])
    max_trials = int(config["DEFAULT"]["max_trials"])
    print("num_cpus: ", num_cpu)
    print("num gpus: ", num_gpu)
    print("max_trials: ", max_trials)
    
    robustnesses_file = config["DEFAULT"]["directory"]+"/robustnesses.csv"
    accuracies_file = config["DEFAULT"]["directory"]+"/accuracies.csv"
    time_file = config["DEFAULT"]["directory"]+"/time.csv"

    # get models and classifiers from json (default: use all supported ones)
    if config["DEFAULT"]["models"] == 'default':
        models = supported_models
    else:
        for model in config["DEFAULT"]["models"].split(", "):
            if model in supported_models:
                models.append(model)
            else:
                print("model \'", model, "\' is not supported")
    if config["DEFAULT"]["classifier"] == 'default':
        classifiers = supported_classifiers
    else:
        for clf in config["DEFAULT"]["classifier"].split(", "):
            if clf in supported_classifiers:
                classifiers.append(clf)
            else:
                print("classifier \'", clf, "\' is not supported")

    print("search space contains the following models: ")
    print(models)
    print("and the following classifiers: ")
    print(classifiers)
    
    def evaluate(classifier, X_val,Y_val):
        y_pred, time_per_prediciton = Prediction_timed(classifier, X_val)
        if not (type(y_pred) == np.ndarray):
           y_pred = np.argmax(y_pred.to("cpu"), axis=1)
        accuracy = accuracy_score(Y_val, y_pred)
        evaluation_dict = {}
        evaluation_dict["accuracy"] = accuracy
        print("accuracy score", accuracy * accuracy_wf)
        robustness1 = robustness_nlp_model(accuracy, classifier, (X_val, Y_val), accuracy_score)
        robustness2 = robustness_noise(y_pred, classifier, (X_val, Y_val), accuracy_score)
        if "unbalanced" in dataset:
            robustness3 = robustness_data_shift(y_pred, classifier, (X_val, Y_val), accuracy_score)
            robustness = (robustness1 + robustness2 + robustness3) /3.0
        else:
            robustness = (robustness1 + robustness2 ) /2.0
#         fairness = evaluate_model_bias(classifier)
#         print("fairness score",  fairness * fairness_wf)
        time_scores = time_sigmoid(time_per_prediciton, wished_time)
        print("robustness score",  robustness * robustness_wf)
        print("time_per_prediciton score", time_scores)
        evaluation_dict["robustness"] = robustness
        evaluation_dict["time_per_prediciton"] = time_per_prediciton
        if search_metric == "custom":
            evaluation_dict[ACTIVE_METRIC_NAME] = accuracy * accuracy_wf + time_scores *time_wf +  robustness * robustness_wf  #+ fairness * fairness_wf
        elif search_metric == "advanced":
            evaluation_dict[ACTIVE_METRIC_NAME] = (score_scaled(accuracy, wished_accuracy)  +  
                                                   time_score(time_per_prediciton, wished_time) +
                                                   score_scaled(robustness, wished_robustness)) #+ min(1,0.1/(DSP_ForeignWorker + 1e-10)) * fairness_wf
        else:
            evaluation_dict[ACTIVE_METRIC_NAME] = accuracy * accuracy_wf 
        print("Overall score:", evaluation_dict[ACTIVE_METRIC_NAME])
        return evaluation_dict
    
    lower = 5e-7
    if "small" in dataset:
        upper = 1e-4
    else:
        upper = 1e-5
    
    def train_fn():
        @ag.args(
                 hidden_fc=ag.space.Int(lower=5, upper=200),
                 lr=ag.space.Real(lower=lower, upper=upper, log=True),
                 model_name=ag.space.Categorical(*models),
                 classifier_type=ag.space.Categorical(*classifiers)
                )
        def run_opaque_box(args, reporter):
            hyperparameters = {}
            hyperparameters["model_name"] = args.model_name
            hyperparameters["hidden_fc"] = args.hidden_fc
            hyperparameters["lr"] = args.lr

            print("current model: ")
            print(args.model_name)
            print("current classifier: ")
            print(args.classifier_type)
            print("current lr: ")
            print(args.lr)
            print("current hidden_fc: ")
            print(args.hidden_fc)
            if args.classifier_type == 'nn':
                nlp_classifier = NLP_embedder(hyperparameters, num_classes, batch_size)
            elif args.classifier_type == 'svm':
                nlp_classifier = hybrid_classifier(hyperparameters, num_classes, batch_size)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nlp_classifier = nlp_classifier.to(device)

            for e in range(max_epochs):
                opaque_box_eval = opaque_box(hyperparameters, nlp_classifier)
                f = open(robustnesses_file, "a")
                f.write(str(opaque_box_eval["robustness"])+ ",")
                f.close()
                f = open(accuracies_file, "a")
                f.write(str(opaque_box_eval["accuracy"])+ ",")
                f.close()
                f = open(time_file, "a")
                f.write(str(opaque_box_eval["time_per_prediciton"])+ ",")
                f.close()
                reporter(objective=opaque_box_eval[ACTIVE_METRIC_NAME], epoch=e)

        return run_opaque_box

    runboxfn = train_fn()
    
    def opaque_box(hyperparameters, classifier):
        
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
#         print(X_train[0:2])
#         print(X_val[0:2])
#         print(Y_train[0:2])
#         print(Y_val[0:2])
        classifier.fit(X_train, Y_train, epochs=1)

        evaluation_dict = evaluate(classifier, X_val, Y_val)
       
        return evaluation_dict
    
    
    # testing
    run_gpt2_test = args.test
    if run_gpt2_test:
        print("run preliminary test")
        print("if you want to run an experiment with the scheduler, deactivate this test to save time")
        hyperparameters = {}
        hyperparameters["model_name"] = 'bert-base-uncased'
        hyperparameters["hidden_fc"] = 100
        hyperparameters["lr"] = 5e-5
        classifier = NLP_embedder(hyperparameters, num_classes, batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = classifier.to(device)
        returndict = opaque_box(hyperparameters, classifier)

        # release all gpu memory (assuming that data was moved to cpu/ deleted during fit and evaluate!)
        classifier = classifier.to('cpu')
        del classifier
        torch.cuda.empty_cache()

        print(returndict)
        print("after preliminary test")

    # Create scheduler and searcher:
    # First get_config are random, the remaining ones use constrained BO
    search_options = {
        'random_seed': SEED,
        'num_fantasy_samples': 5,
        'num_init_random': 1,
        'debug_log': True}

    myscheduler = ag.scheduler.HyperbandScheduler(   
        runboxfn,
        resource={'num_cpus': num_cpu, 'num_gpus': num_gpu},
        searcher='bayesopt',
        search_options=search_options,
        num_trials=max_trials,
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
    
    plotpareto(np.genfromtxt(robustnesses_file, delimiter=',')[:-1], np.genfromtxt(accuracies_file, delimiter=',')[:-1], np.genfromtxt(time_file, delimiter=',')[:-1], config["DEFAULT"]["directory"]+"/pareto.png")
    




