import sys
import os
import pandas as pd
sys.path.insert(1, r'experiments\Standardized Pipeline')

import Finalized_pipeline


regression = False

if regression:
    param_grid_dt={
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        'splitter': ["best", "random"]
    }
    param_grid_rf={
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_grid_lr = { ### Regression
    }
    param_grid_nn = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500, 1000]
    }
    param_grid_gb={
        'n_estimators': [10, 100, 200], 
        'learning_rate': [0.1,0.5,1.0,2.0],
        'max_depth': [3, 5, 7]
    }
    param_grid_xg = {
        'eta': [0.3, 0.4, 0.5],
        'gamma': [0, 0.01, 0.1]
    }
    param_grid_sv = { ### regression
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ["rbf", "linear", "poly", "sigmoid"],
        'epsilon': [0.01, 0.1, 1],
        'gamma': ["scale", 0.1, 0.05]
    }

else:
    param_grid_dt={
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_grid_rf={
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    param_grid_nn = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500, 1000]
    }
    param_grid_gb={
        'n_estimators': [10, 100, 200], 
        'learning_rate': [0.1,0.5], #,1.0,2.0],
        'max_depth': [3, 5, 7]
    }
    param_grid_xg = {
        'eta': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'gamma': [0, 0.01, 0.1]
    }
    param_grid_sv = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ["rbf"], #, "linear", "poly", "sigmoid"],
        'gamma': ["scale", 0.1, 0.05]
    }


preprocessed_dataset_path = r"experiments\Standardized Pipeline\pipeline_example\preprocessed_dataset_descriptor.csv"

if not os.path.exists(preprocessed_dataset_path):
    df = Finalized_pipeline.generate_split_dataset(r"experiments\ROR-gamma\ROR_data_1.csv")
    print(df.head())
    
    df = Finalized_pipeline.calculate_pIC50(df, "target")
    print(df.head())
    df = Finalized_pipeline.calculate_classification_labels(df, "pIC50", threshold=7)
    print(df.head())

    if regression:
        target_column = "pIC50"
    else:
        target_column = "label"

    df = Finalized_pipeline.calculate_features(df, calculate_descriptors=True, calculate_fingerprints=False, 
                                                        SMILES_column_name="SMILES", target_column_name=target_column, 
                                                        split_column_name="Split", output_csv_path=preprocessed_dataset_path)
    print(df.head())

hyperparams = {"dt": param_grid_dt, "rf": param_grid_rf, "lr": param_grid_lr, "nn": param_grid_nn, "gb": param_grid_gb, "xg": param_grid_xg, "sv": param_grid_sv}
hyperparams = {"xg": param_grid_xg}
hyperparams = {"sv": param_grid_sv}
Finalized_pipeline.hyperparameter_search(preprocessed_dataset_path, hyperparams, unique=True)