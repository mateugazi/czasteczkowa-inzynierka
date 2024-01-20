import os
import pandas as pd
import Finalized_pipeline

name = "Asia"
#name = "Kuba"
#name = "Wojtek"

if name == "Asia":
    calculate_descriptors = True
    calculate_fingerprints = False
if name == "Kuba":
    calculate_descriptors = True
    calculate_fingerprints = True
if name == "Wojtek":
    calculate_descriptors = False
    calculate_fingerprints = True

for regression in [True]:
    if regression:
        param_grid_dt={
            'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
            'min_impurity_decrease': [0, 0.05, 0.1, 0.2], 'splitter': ["best", "random"],
            'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"]
        }
        param_grid_rf={
            'max_depth': [None, 3, 7, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
            'min_impurity_decrease': [0, 0.05, 0.1, 0.2], 'n_estimators': [10, 50, 100, 200]
        }
        param_grid_lr = { ### Regression
        }
        param_grid_nn = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01], 'max_iter': [200, 500, 1000], 'solver': ['lbfgs', 'sgd', 'adam']
        }
        param_grid_gb={ # 3, 
            'max_depth': [3, 5, 7], 'loss': ['squared_error', 'absolute_error', 'huber'], #, 'quantile'],
            'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5, 1.0], # , 2.0],
            'min_impurity_decrease': [0, 0.1, 0.2] # [0, 0.05, 0.1, 0.2]
        }
        param_grid_xg = {
            'max_depth': [None, 3, 5, 7], 'eta': [0.01, 0.1, 0.2], 'gamma': [0, 0.01, 0.1]
        }
        param_grid_sv = { ### regression
            'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ["rbf", "poly", "sigmoid"], # , "linear"
            'epsilon': [0.01, 0.1, 1], 'gamma': ["scale", 0.1, 0.05]
        }

    else:
        param_grid_dt={
            'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
            'min_impurity_decrease': [0, 0.05, 0.1, 0.2], 'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ["best", "random"]
        }
        param_grid_rf={
            'max_depth': [None, 3, 7, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
            'min_impurity_decrease': [0, 0.05, 0.1, 0.2], 'n_estimators': [10, 50, 100, 200]
        }
        param_grid_lr = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']
        }
        param_grid_nn = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01], 'max_iter': [200, 500, 1000], 'solver': ['lbfgs', 'sgd', 'adam']
        }
        param_grid_gb={
            'max_depth': [None, 3, 5, 7], 'loss': ['log_loss', 'exponential'], 'n_estimators': [10, 50, 100, 200], 
            'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0], 'min_impurity_decrease': [0, 0.05, 0.1, 0.2]
        }
        param_grid_xg = {
            'max_depth': [None, 3, 5, 7], 'eta': [0.01, 0.1, 0.2], 'gamma': [0, 0.01, 0.1]
        }
        param_grid_sv = { ### classification
            'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ["rbf", "linear", "poly", "sigmoid"], 
            'gamma': ["scale", 0.1, 0.05]
        }

    
    datasets = []
    for (dirpath, dirnames, filenames) in os.walk(r"experiments\split_datasets"):
        datasets.extend(filenames)
        break


    for dataset in datasets[0:1]:
        ### Check which files will be used:
        #print(dataset)
        #continue

        ### CHANGE SV to not run C 1000 with linear kernel!
        dataset_path = os.path.join(r"experiments\split_datasets", dataset)

        df = pd.read_csv(dataset_path)

        if not "pIC50" in df.columns:
            df = Finalized_pipeline.calculate_pIC50(df, "target")
        
        if not regression:
            df = Finalized_pipeline.calculate_classification_labels(df, "pIC50", threshold=7)
            target_column = "label"
        else:
            target_column = "pIC50"

        if "mol" in df.columns:
            df.rename(columns={"mol": "SMILES"}, inplace=True)
            
        if regression:
            runtype = "regression"
        else:
            runtype = "classification"
        
        df = Finalized_pipeline.calculate_features(df, calculate_descriptors=calculate_descriptors, calculate_fingerprints=calculate_fingerprints, 
                                                        SMILES_column_name="SMILES", target_column_name=target_column, 
                                                        split_column_name="Split", output_csv_path=r"experiments\Standardized Pipeline\example_dataset.csv")
        
        hyperparams = {"gb": {}}
        Finalized_pipeline.hyperparameter_search(df, hyperparams, output_file_name=name + "_" + dataset[:13] + "_" + runtype + "_run.csv")
