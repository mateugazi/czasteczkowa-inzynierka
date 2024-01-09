import pandas as pd
import numpy as np
import os
import datetime
from itertools import product
import pickle

from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score

from sklearn import tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from xgboost import XGBClassifier, XGBRegressor


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

### Input standard SMILES column
def CalculateMorganFingerprint(mol):
    mol = mol.apply(Chem.MolFromSmiles)
    mfpgen = AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    fingerprint = np.array([mfpgen.GetFingerprintAsNumPy(x) for x in mol])
    fingerprint = pd.DataFrame(fingerprint, columns = ['mfp'+str(i) for i in range(fingerprint.shape[1])])
    return fingerprint

### Input standard SMILES column
def CalculateDescriptors(mol):
    mol = mol.apply(Chem.MolFromSmiles)
    calc = Calculator(descriptors, ignore_3D=False)
    X_mordred = calc.pandas(mol, nproc=1)
    X_mordred = X_mordred.select_dtypes(['number'])
    #normalize
    X_mordred = (X_mordred-X_mordred.min())/(X_mordred.max()-X_mordred.min())
    #drop columns wth low std
    X_mordred = X_mordred.loc[:,X_mordred.std()>0.01]
    return X_mordred


def generate_split_dataset(path, train_fraction_split=0.8):
    test_fraction_split = 1 - train_fraction_split
    df = pd.read_csv(path)   

    ### Generate Split column
    train, test = train_test_split(df, test_size=test_fraction_split, random_state=42)
    train.insert(0, 'Split', 'train')
    test.insert(0, 'Split', 'test')
    df = pd.concat([train, test], axis=0)

    output_path = r"experiments\split_datasets\\split" + str(1 - test_fraction_split) + "_" + os.path.basename(path)
    df.to_csv(output_path)
    print(output_path)
    return output_path

def Load_downloaded_CSV(path, use_descriptors, use_fingerprints, regression, calculate_pIC50 = False, threshold=7, split_column="Split"):
    df = pd.read_csv(path)
    
    ### Replace with standardizing molecules and then dropping duplicates
    #df.drop_duplicates('mol')
    #df = df.dropna()

    ### Check if the df has a split column already, if not, generate split column
    if split_column in df.columns:
        ### TODO: Check if only values in split are 'train' and 'test'?
        if split_column != "Split":
            df['Split'] = df[split_column]
            df.drop(split_column, axis=1, inplace=True)
    else:
        new_path = generate_split_dataset(path)
        df = pd.read_csv(new_path)


    df.rename(columns={"target": "Target", "SMILES": 'mol'}, inplace=True)
        
    if regression:
        if 'IC50' in df.columns:
            calculate_pIC50 = True
            df['Target'] = df['IC50']
            df.drop('IC50', axis=1, inplace=True)
            
        if 'pIC50' in df.columns:
            df['Target'] = df['pIC50']
            df.drop('pIC50', axis=1, inplace=True)
    else:
        df.rename(columns={"Class": "Target"}, inplace=True)
        
    
    if calculate_pIC50:
        df['Target'] = [-np.log10(i * 10**(-9)) for i in list(df['Target'])]
        if not regression:
            df['Target'] = [int(i > threshold) for i in list(df['Target'])]

    df = df[['mol', 'Target', 'Split']]

    if use_descriptors:
        new_df = CalculateDescriptors(df['mol'])
    if use_fingerprints:
        new_df = CalculateMorganFingerprint(df['mol'])
        
    new_df['Target'] = df['Target']
    new_df['Split'] = df['Split']

    train = new_df[new_df['Split'] == 'train']
    test = new_df[new_df['Split'] == 'test']

    X_train = train.drop(['Target', 'Split'], axis=1)
    y_train = train[['Target']]
    X_test = test.drop(['Target', 'Split'], axis=1)
    y_test = test[['Target']]

    return X_train, y_train, X_test, y_test

#def Split_downloaded_CSV(df):
#    X = df.drop(['Target'], axis=1)
#    y = df[['Target']]
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#    return X_train, y_train, X_test, y_test


def model_builder(model_name, hyperparams, regression):
    if model_name == 'dt':
        if "min_samples_split" not in hyperparams.keys():
            hyperparams["min_samples_split"] = 2
        if "max_depth" not in hyperparams.keys():
            hyperparams["max_depth"] = None  

        if regression:
            if "criterion" not in hyperparams.keys():
                hyperparams["criterion"] = "squared_error"
            print(hyperparams)
            model = tree.DecisionTreeRegressor(max_depth=hyperparams["max_depth"],
                                    min_samples_split=hyperparams["min_samples_split"],
                                    criterion=hyperparams["criterion"])
        else:
            if "criterion" not in hyperparams.keys():
                hyperparams["criterion"] = "gini"
            print(hyperparams)
            model = tree.DecisionTreeClassifier(max_depth=hyperparams["max_depth"],
                                    min_samples_split=hyperparams["min_samples_split"], 
                                    criterion=hyperparams["criterion"])
    if model_name == 'rf':
        if "n_estimators" not in hyperparams.keys():
            hyperparams["n_estimators"] = 100
        if "min_samples_split" not in hyperparams.keys():
            hyperparams["min_samples_split"] = 2
        if "bootstrap" not in hyperparams.keys():
            hyperparams["bootstrap"] = True  
        if "max_depth" not in hyperparams.keys():
            hyperparams["max_depth"] = None  

        if regression:
            if "criterion" not in hyperparams.keys():
                hyperparams["criterion"] = "squared_error"
            print(hyperparams)
            model = RandomForestRegressor(n_estimators=hyperparams["n_estimators"],
                                    max_depth=hyperparams["max_depth"],
                                    min_samples_split=hyperparams["min_samples_split"],
                                    criterion=hyperparams["criterion"],
                                    bootstrap=hyperparams["bootstrap"])
        else:
            if "criterion" not in hyperparams.keys():
                hyperparams["criterion"] = "gini"
            print(hyperparams)
            model = RandomForestClassifier(n_estimators=hyperparams["n_estimators"],
                                    max_depth=hyperparams["max_depth"],
                                    min_samples_split=hyperparams["min_samples_split"], 
                                    criterion=hyperparams["criterion"],
                                    bootstrap=hyperparams["bootstrap"])
            
    if model_name == 'lr':
        if regression:
            print("no hyperparameters")
            model = LinearRegression()
        else:
            if "C" not in hyperparams.keys():
                hyperparams["C"] = 1
            if "penalty" not in hyperparams.keys():
                hyperparams["penalty"] = "l2"
            if "solver" not in hyperparams.keys():
                hyperparams["solver"] = "liblinear"
            print(hyperparams)
            model = LogisticRegression(C=hyperparams["C"], 
                                       penalty=hyperparams["penalty"], 
                                       solver=hyperparams["solver"])

    if model_name == 'nn':
        if "hidden_layer_sizes" not in hyperparams.keys():
            hyperparams["hidden_layer_sizes"] = (100,)
        if "activation" not in hyperparams.keys():
            hyperparams["activation"] = "relu"
        if "alpha" not in hyperparams.keys():
            hyperparams["alpha"] = 0.0001  
        if "max_iter" not in hyperparams.keys():
            hyperparams["max_iter"] = 500#200
        if regression:
            print(hyperparams)
            model = MLPRegressor(hidden_layer_sizes=hyperparams["hidden_layer_sizes"], 
                                activation=hyperparams["activation"], 
                                alpha=hyperparams["alpha"], max_iter=hyperparams["max_iter"])
        else:
            print(hyperparams)
            model = MLPClassifier(hidden_layer_sizes=hyperparams["hidden_layer_sizes"], 
                                activation=hyperparams["activation"], 
                                alpha=hyperparams["alpha"], max_iter=hyperparams["max_iter"])
        
    if model_name == 'gb':
        if "n_estimators" not in hyperparams.keys():
            hyperparams["n_estimators"] = 100
        if "learning_rate" not in hyperparams.keys():
            hyperparams["learning_rate"] = 0.1
        if regression:
            print(hyperparams)
            model = GradientBoostingRegressor(n_estimators=hyperparams["n_estimators"], 
                                              learning_rate=hyperparams["learning_rate"])
        else:
            print(hyperparams)
            model = GradientBoostingClassifier(n_estimators=hyperparams["n_estimators"], 
                                               learning_rate=hyperparams["learning_rate"])

    if model_name == 'xg':
        #if "C" not in hyperparams.keys():
        #    hyperparams["C"] = 1
        #if "degree" not in hyperparams.keys():
        #    hyperparams["degree"] = 3
        #if "kernel" not in hyperparams.keys():
        #    hyperparams["kernel"] = "rbf"
        if regression:
            model = XGBRegressor()
        else:
            model = XGBRegressor()
            
    if model_name == 'sv':
        if "C" not in hyperparams.keys():
            hyperparams["C"] = 1
        if "degree" not in hyperparams.keys():
            hyperparams["degree"] = 3
        if "kernel" not in hyperparams.keys():
            hyperparams["kernel"] = "rbf"
        if regression:
            if "epsilon" not in hyperparams.keys():
                hyperparams["epsilon"] = 0.1
            model = SVR(C=hyperparams["C"], 
                        degree=hyperparams["degree"], 
                        kernel=hyperparams["kernel"], 
                        epsilon=hyperparams["epsilon"])
        else:
            model = SVC(C=hyperparams["C"], 
                        degree=hyperparams["degree"], 
                        kernel=hyperparams["kernel"])
            
    return model


def train_and_test(model, X_train, y_train, X_test, y_test, regression, metrics=[], iterations=1):
    for i in range(iterations):
        ### TODO: Info if IC should've been calculated
        model.fit(X_train, np.reshape(y_train, (-1, )))
        
        y_test_predicted = model.predict(X_test)
        y_test_predicted = list(map(round, y_test_predicted))

        #print("Standard train-test results:")

        results_test = {}

        if regression:
            if 'rmse' in metrics or len(metrics) == 0:
                metric_test = mean_squared_error(y_test, y_test_predicted, squared=False)
                results_test["rmse"] = metric_test
            if 'mse' in metrics or len(metrics) == 0:
                metric_test = mean_squared_error(y_test, y_test_predicted)
                results_test["mse"] = metric_test
            if 'mae' in metrics or len(metrics) == 0:
                metric_test = mean_absolute_error(y_test, y_test_predicted)
                results_test["mae"] = metric_test
            if 'r2' in metrics or len(metrics) == 0:
                metric_test = r2_score(y_test, y_test_predicted)
                results_test["r2"] = metric_test
            
        else:
            if 'roc_auc' in metrics or len(metrics) == 0:
                metric_test = roc_auc_score(y_test, y_test_predicted)
                results_test["roc_auc"] = metric_test
            if 'accuracy' in metrics or len(metrics) == 0:
                metric_test = accuracy_score(y_test, y_test_predicted)
                results_test["accuracy"] = metric_test
            if 'precision' in metrics or len(metrics) == 0:
                metric_test = precision_score(y_test, y_test_predicted)
                results_test["precision"] = metric_test
            if 'recall' in metrics or len(metrics) == 0:
                metric_test = recall_score(y_test, y_test_predicted)
                results_test["recall"] = metric_test
            if 'f1' in metrics or len(metrics) == 0:
                metric_test = f1_score(y_test, y_test_predicted)
                results_test["f1"] = metric_test

    return results_test

### All hyperparameters need to be supplimented into a function
def pipeline(csv_path, regression, dt_parameters, rf_parameters, lr_parameters, nn_parameters, gb_parameters, xg_parameters, sv_parameters, split_column="Split", calculate_pIC50=False, pIC50_classification_threshold=7, output_path="results.csv"):
    model_name_dict_reg = {"dt": "DecisionTreeRegressor", "rf": "RandomForestRegressor", "lr": "LinearRegression", "nn": "MLPRegressor", "gb": "GradientBoostingRegressor", "xg": "XGBRegressor", "sv": "SVR"}
    model_name_dict_class = {"dt": "DecisionTreeClassifier", "rf": "RandomForestClassifier", "lr": "LogisticRegression", "nn": "MLPClassifier", "gb": "GradientBoostingClassifier", "xg": "XGBClassifier", "sv": "SVC"}

    ### TODO: Change into args!
    use_descriptors = True
    use_fingerprints = False

    models_with_parameters_to_separate = [
    ['dt', dt_parameters],
    ['rf', rf_parameters],
    ['lr', lr_parameters],
    ['nn', nn_parameters],
    ['gb', gb_parameters],
    ['xg', xg_parameters],
    ['sv', sv_parameters]]

    models_with_parameters = []

    for model_name, parameter_grid in models_with_parameters_to_separate:
        keys = parameter_grid.keys()
        param_combinations = list(product(*parameter_grid.values()))
        for combination in param_combinations:
            param_set = {}
            for index, k in enumerate(keys):
                param_set[k] = combination[index]
            models_with_parameters.append([model_name, param_set])

    metrics = []
    
    ### ror-gamma
    #csv_path = r"C:\Users\admin\Documents\GitHub\czasteczkowa-inzynierka\experiments\ROR-gamma\ROR_data_1.csv"
    #df_loaded = Load_downloaded_CSV(csv_path, regression=regression, calculate_pIC50=True)
    ### bace
    #csv_path = r"C:\Users\admin\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv"
    #df_loaded = Load_downloaded_CSV(csv_path, regression=regression, use_descriptors=use_descriptors, use_fingerprints=use_fingerprints)

    print("Calculating input data")
    begin = datetime.datetime.now()
    
    X_train, y_train, X_test, y_test = Load_downloaded_CSV(csv_path, regression=regression, calculate_pIC50=calculate_pIC50, threshold=pIC50_classification_threshold,
                                    use_descriptors=use_descriptors, use_fingerprints=use_fingerprints, split_column=split_column)
    
    elapsed = (datetime.datetime.now() - begin).total_seconds()
    print(elapsed)
    

    data = {"model": []}
    if regression:
        for metric in ["mse", "rmse", "mae", "r2"]:
            data[metric] = []
    else:
        for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            data[metric] = []

    results_df = pd.DataFrame(data)

    best_model = None
    if regression:
        compared_score = "mse"
        best_model_score = 100000
    else:
        compared_score = "roc_auc"
        best_model_score = 0

    output_path = uniquify(output_path)


    #
    f = open(output_path, "w")
    f.write(",".join(data.keys()))
    f.write("\n")

    i = -1
    for model_name, hyperparams in models_with_parameters:
        i += 1
        print(model_name)

        model = model_builder(model_name, hyperparams, regression)
        
        begin = datetime.datetime.now()
        
        results_test = train_and_test(model, X_train, y_train, X_test, y_test, regression=regression, metrics=metrics)
        
        elapsed = (datetime.datetime.now() - begin).total_seconds()
        print(elapsed)
        
        if regression:
            results_test["model"] = model_name_dict_reg[model_name]
        else:
            results_test["model"] = model_name_dict_class[model_name]
        results_df.loc[len(results_df)] = results_test

        ### File write results
        f.write(",".join([str(i)] + [str(i) for i in list(results_test.values())]))
        f.write("\n")

        if regression and results_test[compared_score] < best_model_score:
            best_model_score = results_test[compared_score]
            best_model = model
        if not regression and results_test[compared_score] > best_model_score:
            best_model_score = results_test[compared_score]
            best_model = model

    f.close()
    results_df.to_csv(output_path)

    filename = os.path.join(os.path.dirname(output_path), 'model.sav')
    pickle.dump(best_model, open(filename, 'wb'))

