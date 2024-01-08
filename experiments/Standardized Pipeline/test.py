import Finalized_pipeline

regression = False

if regression:
    param_grid_rf={
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_grid_lr = { ### Regression
        'C': [0.001],
        'penalty': ['l1'],
        'solver': ['liblinear']
    }
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500, 1000]
    }
    param_grid_gb={
        'n_estimators': [10, 100, 200], 
        'learning_rate': [0.1,0.5,1.0,2.0]
    }
    param_grid_svm = { ### regression
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'degree': [2, 3, 4],
        'epsilon': [0.01, 0.1, 1]
    }

else:
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
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500, 1000]
    }
    param_grid_gb={
        'n_estimators': [10, 100, 200], 
        'learning_rate': [0.1,0.5,1.0,2.0]
    }
    param_grid_svm = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'degree': [2, 3, 4, 5],
        'epsilon': ["no epsilon"]
    }


Finalized_pipeline.pipeline(r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv",
                            regression, {}, {}, 
                            {}, {}, {}, 
                            output_path=r"C:\Users\wojci\Documents\GitHub\czasteczkowa-inzynierka\experiments\Standardized Pipeline\results2.csv")