import Finalized_pipeline
import Separate_pipeline_functions

import os
regression = True

if regression:
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
    }
    param_grid_sv = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ["rbf"], #, "linear", "poly", "sigmoid"],
        'gamma': ["scale", 0.1, 0.05]
    }


#out_path = r"experiments\Standardized Pipeline\example_descriptors.csv"
#df = Separate_pipeline_functions.generate_split_dataset(r"experiments\ROR-gamma\ROR_data_1.csv")
#print(df.head())
#df = Separate_pipeline_functions.calculate_pIC50(df, "target")
#print(df.head())
#df = Separate_pipeline_functions.calculate_classification_labels(df, "pIC50", threshold=7)
#print(df.head())
#df = Separate_pipeline_functions.calculate_features(df, calculate_descriptors=True, calculate_fingerprints=False, SMILES_column_name="SMILES", target_column_name="pIC50", split_column_name="Split", output_csv_path=out_path)
#print(df.head())
#
#hyperparams = {"dt": param_grid_dt, "rf": param_grid_rf, "lr": param_grid_lr, "nn": param_grid_nn, "gb": param_grid_gb, "xg": param_grid_xg, "sv": param_grid_sv}
#Separate_pipeline_functions.hyperparameter_search(df, hyperparams)


hyperparams = {"dt": param_grid_dt, "rf": param_grid_rf, "lr": param_grid_lr, "nn": param_grid_nn, "gb": param_grid_gb, "xg": param_grid_xg, "sv": param_grid_sv}
Separate_pipeline_functions.hyperparameter_search(r"experiments\Standardized Pipeline\example_descriptors.csv", hyperparams)

#df = Separate_pipeline_functions.hyperparameter_search(r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv",
#                            #r"experiments\split_datasets\split0.8_bace.csv",
#                            r"experiments\split_datasets\split0.9_ROR_data_1.csv",
#                            regression, param_grid_dt, param_grid_rf, param_grid_lr,
#                            param_grid_nn, param_grid_gb, param_grid_xg, param_grid_sv, 
#                            output_path=r"experiments\Standardized Pipeline\RESULTS_ROR0.9_classification.csv",
#                            calculate_pIC50=True)
#
#print(df.head())


#df = Separate_pipeline_functions.calculate_features(df, "target")
#print(df.head())


#Finalized_pipeline.generate_split_dataset(r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv", train_fraction_split=0.9)
#Finalized_pipeline.generate_split_dataset(r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv")
#Finalized_pipeline.generate_split_dataset(r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv", train_fraction_split=0.7)
#Finalized_pipeline.generate_split_dataset(r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv", train_fraction_split=0.6)
#
#Finalized_pipeline.generate_split_dataset(r"experiments\ROR-gamma\ROR_data_1.csv", train_fraction_split=0.9)
#Finalized_pipeline.generate_split_dataset(r"experiments\ROR-gamma\ROR_data_1.csv")
#Finalized_pipeline.generate_split_dataset(r"experiments\ROR-gamma\ROR_data_1.csv", train_fraction_split=0.7)
#Finalized_pipeline.generate_split_dataset(r"experiments\ROR-gamma\ROR_data_1.csv", train_fraction_split=0.6)

#Finalized_pipeline.pipeline(#r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv",
#                            #r"experiments\split_datasets\split0.8_bace.csv",
#                            r"experiments\split_datasets\split0.8_ROR_data_1.csv",
#                            regression, param_grid_dt, param_grid_rf, param_grid_lr,
#                            param_grid_nn, param_grid_gb, param_grid_xg, param_grid_sv, 
#                            output_path=r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\Standardized Pipeline\RESULTS_ROR_classification.csv",
#                            calculate_pIC50=True)

#### Just svc
#Finalized_pipeline.pipeline(#r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv",
#                            #r"experiments\split_datasets\split0.8_bace.csv",
#                            r"experiments\split_datasets\split0.9_ROR_data_1.csv",
#                            regression, {}, {}, {},
#                            {}, {}, {}, param_grid_sv, 
#                            output_path=r"experiments\Standardized Pipeline\RESULTS_ROR0.9_classification.csv",
#                            calculate_pIC50=True)

#Finalized_pipeline.pipeline(#r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv",
#                            #r"experiments\split_datasets\split0.8_bace.csv",
#                            r"experiments\split_datasets\split0.9_ROR_data_1.csv",
#                            regression, param_grid_dt, param_grid_rf, param_grid_lr,
#                            param_grid_nn, param_grid_gb, param_grid_xg, param_grid_sv, 
#                            output_path=r"experiments\Standardized Pipeline\RESULTS_ROR0.9_classification.csv",
#                            calculate_pIC50=True)