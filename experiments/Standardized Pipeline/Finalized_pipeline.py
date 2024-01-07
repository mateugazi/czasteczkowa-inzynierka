import pandas as pd
import numpy as np
import os
import datetime

from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC


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

def Load_downloaded_CSV(path, use_descriptors, use_fingerprints, regression, calculate_pIC50 = False, threshold=7):
    df = pd.read_csv(path)
    
    ### Replace with standardizing molecules and then dropping duplicates
    #df.drop_duplicates('mol')
    #df = df.dropna()
    
    if 'target' in df.columns:
        df['Target'] = df['target']
        df.drop('target', axis=1, inplace=True)
        
    if regression:
        if 'IC50' in df.columns:
            calculate_pIC50 = True
            df['Target'] = df['IC50']
            df.drop('IC50', axis=1, inplace=True)
        if 'pIC50' in df.columns:
            df['Target'] = df['pIC50']
            df.drop('pIC50', axis=1, inplace=True)
    else:
        if 'Class' in df.columns:
            df['Target'] = df['Class']
            df.drop('Class', axis=1, inplace=True)
        
    if 'SMILES' in df.columns:
        df['mol'] = df['SMILES']
        df.drop('SMILES', axis=1, inplace=True)
    
    if calculate_pIC50:
        df['Target'] = [-np.log10(i * 10**(-9)) for i in list(df['Target'])]
        if not regression:
            df['Target'] = [int(i > threshold) for i in list(df['Target'])]

    df = df[['mol', 'Target']]

    if use_descriptors:
        new_df = CalculateDescriptors(df['mol'])
    if use_fingerprints:
        new_df = CalculateMorganFingerprint(df['mol'])
        
    new_df['Target'] = df['Target']

    return new_df

def Split_downloaded_CSV(df):
    X = df.drop(['Target'], axis=1)
    y = df[['Target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.111, random_state=42)
    return X_train, y_train, X_test, y_test, X_valid, y_valid


def model_builder(model_name, hyperparams, regression):
    if model_name == 'rf':
        if "n_estimators" not in hyperparams.keys():
            hyperparams["n_estimators"] = 100
        if "min_samples_split" not in hyperparams.keys():
            hyperparams["min_samples_split"] = 2
        if "bootstrap" not in hyperparams.keys():
            hyperparams["bootstrap"] = True  

        if regression:
            if "criterion" not in hyperparams.keys():
                hyperparams["criterion"] = "squared_error"
            model = RandomForestRegressor(n_estimators=hyperparams["n_estimators"],
                                    min_samples_split=hyperparams["min_samples_split"],
                                    criterion=hyperparams["criterion"],
                                    bootstrap=hyperparams["bootstrap"])
        else:
            if "criterion" not in hyperparams.keys():
                hyperparams["criterion"] = "gini"
            model = RandomForestClassifier(n_estimators=hyperparams["n_estimators"],
                                    min_samples_split=hyperparams["min_samples_split"], 
                                    criterion=hyperparams["criterion"],
                                    bootstrap=hyperparams["bootstrap"])
            
    if model_name == 'lr':
        if regression:
            model = LinearRegression()
        else:
            if "C" not in hyperparams.keys():
                hyperparams["C"] = 1
            if "penalty" not in hyperparams.keys():
                hyperparams["penalty"] = "l2"
            if "solver" not in hyperparams.keys():
                hyperparams["solver"] = "liblinear"
            model = LogisticRegression(C=hyperparams["C"], penalty=hyperparams["penalty"], solver=hyperparams["solver"])

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
            model = MLPRegressor(hidden_layer_sizes=hyperparams["hidden_layer_sizes"], activation=hyperparams["activation"], 
                                  alpha=hyperparams["alpha"], max_iter=hyperparams["max_iter"])
        else:
            model = MLPClassifier(hidden_layer_sizes=hyperparams["hidden_layer_sizes"], activation=hyperparams["activation"], 
                                  alpha=hyperparams["alpha"], max_iter=hyperparams["max_iter"])
        
    if model_name == 'gb':
        if "n_estimators" not in hyperparams.keys():
            hyperparams["n_estimators"] = 100
        if "learning_rate" not in hyperparams.keys():
            hyperparams["learning_rate"] = 0.1
        if regression:
            model = GradientBoostingRegressor(n_estimators=hyperparams["n_estimators"], learning_rate=hyperparams["learning_rate"])
        else:
            model = GradientBoostingClassifier(n_estimators=hyperparams["n_estimators"], learning_rate=hyperparams["learning_rate"])

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
            model = SVR(C=hyperparams["C"], degree=hyperparams["degree"], kernel=hyperparams["kernel"], epsilon=hyperparams["epsilon"])
        else:
            model = SVC(C=hyperparams["C"], degree=hyperparams["degree"], kernel=hyperparams["kernel"])
            
    return model


def train_and_test(model, X_train, y_train, X_test, y_test, X_valid, y_valid, regression, metrics=[], iterations=1):
    for i in range(iterations):
        model.fit(X_train, np.reshape(y_train, (-1, )))
        
        y_test_predicted = model.predict(X_test)
        y_valid_predicted = model.predict(X_valid)

        #print("Standard train-test results:")

        results_test = {}
        results_valid = {}

        if regression:
            if 'rmse' in metrics or len(metrics) == 0:
                metric_test = mean_squared_error(y_test, y_test_predicted, squared=False)
                metric_valid = mean_squared_error(y_valid, y_valid_predicted, squared=False)
                results_test["rmse"] = metric_test
                results_valid["rmse"] = metric_valid
            if 'mse' in metrics or len(metrics) == 0:
                metric_test = mean_squared_error(y_test, y_test_predicted)
                metric_valid = mean_squared_error(y_valid, y_valid_predicted)
                results_test["mse"] = metric_test
                results_valid["mse"] = metric_valid
            if 'mae' in metrics or len(metrics) == 0:
                metric_test = mean_absolute_error(y_test, y_test_predicted)
                metric_valid = mean_absolute_error(y_valid, y_valid_predicted)
                results_test["mae"] = metric_test
                results_valid["mae"] = metric_valid
            if 'r2' in metrics or len(metrics) == 0:
                metric_test = r2_score(y_test, y_test_predicted)
                metric_valid = r2_score(y_valid, y_valid_predicted)
                results_test["r2"] = metric_test
                results_valid["r2"] = metric_valid
            
        else:
            if 'roc_auc' in metrics or len(metrics) == 0:
                metric_test = roc_auc_score(y_test, y_test_predicted)
                metric_valid = roc_auc_score(y_valid, y_valid_predicted)
                results_test["roc_auc"] = metric_test
                results_valid["roc_auc"] = metric_valid
            if 'accuracy' in metrics or len(metrics) == 0:
                metric_test = accuracy_score(y_test, y_test_predicted)
                metric_valid = accuracy_score(y_valid, y_valid_predicted)
                results_test["accuracy"] = metric_test
                results_valid["accuracy"] = metric_valid
            if 'precision' in metrics or len(metrics) == 0:
                metric_test = precision_score(y_test, y_test_predicted)
                metric_valid = precision_score(y_valid, y_valid_predicted)
                results_test["precision"] = metric_test
                results_valid["precision"] = metric_valid
            if 'recall' in metrics or len(metrics) == 0:
                metric_test = recall_score(y_test, y_test_predicted)
                metric_valid = recall_score(y_valid, y_valid_predicted)
                results_test["recall"] = metric_test
                results_valid["recall"] = metric_valid
            if 'f1' in metrics or len(metrics) == 0:
                metric_test = f1_score(y_test, y_test_predicted)
                metric_valid = f1_score(y_valid, y_valid_predicted)
                results_test["f1"] = metric_test
                results_valid["f1"] = metric_valid

    return results_test, results_valid

### All hyperparameters need to be supplimented into a function
def pipeline(csv_path, regression, rf_parameters, lr_parameters, nn_parameters, gb_parameters, sv_parameters, calculate_pIC50=False, pIC50_classification_threshold=7, output_path="results.csv"):
    model_name_dict_reg = {"rf": "RandomForestRegressor", "lr": "LinearRegression", "nn": "MLPRegressor", "gb": "GradientBoostingRegressor", "sv": "SVR"}
    model_name_dict_class = {"rf": "RandomForestClassifier", "lr": "LogisticRegression", "nn": "MLPClassifier", "gb": "GradientBoostingClassifier", "sv": "SVC"}

    ### TODO: Change into args!
    use_descriptors = True
    use_fingerprints = False

    regression = True

    #models_with_parameters_pending_split = [
    models_with_parameters = [
    ['rf', rf_parameters],
    ['lr', lr_parameters],
    ['nn', nn_parameters],
    ['gb', gb_parameters],
    ['sv', sv_parameters]]
    
    #models_with_parameters = []
    #for row in models_with_parameters_pending_split:
    #    parameter_dict = row[1]
    #    static_params = {}
    #    param_ranges = {}
    #    
    #    for key, values in parameter_dict.items():
    #        if len(values) == 1:
    #            static_params[key] = values
    #        if len(values) >= 1:
    #            param_ranges[key] = values
    #            
        ### TODO: make all combinations of parameters
                
        

    metrics = []
    
    ### ror-gamma
    #csv_path = r"C:\Users\admin\Documents\GitHub\czasteczkowa-inzynierka\experiments\ROR-gamma\ROR_data_1.csv"
    #df_loaded = Load_downloaded_CSV(csv_path, regression=regression, calculate_pIC50=True)
    ### bace
    #csv_path = r"C:\Users\admin\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv"
    #df_loaded = Load_downloaded_CSV(csv_path, regression=regression, use_descriptors=use_descriptors, use_fingerprints=use_fingerprints)

    print("Calculating input data")
    begin = datetime.datetime.now()
    
    df_loaded = Load_downloaded_CSV(csv_path, regression=regression, calculate_pIC50=calculate_pIC50, threshold=pIC50_classification_threshold,
                                    use_descriptors=use_descriptors, use_fingerprints=use_fingerprints)
    
    elapsed = (datetime.datetime.now() - begin).total_seconds()
    print(elapsed)
    
    X_train, y_train, X_test, y_test, X_valid, y_valid = Split_downloaded_CSV(df_loaded)

    data = {"model": [], "set": []}
    if regression:
        for metric in ["rmse", "mse", "mae", "r2"]:
            data[metric] = []
    else:
        for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            data[metric] = []

    results_df = pd.DataFrame(data)
    for model_name, hyperparams in models_with_parameters:
        print(model_name)

        model = model_builder(model_name, hyperparams, regression)
        
        begin = datetime.datetime.now()
        
        results_test, results_valid = train_and_test(model, X_train, y_train, X_test, y_test, X_valid, y_valid, regression=regression, metrics=metrics)
        
        elapsed = (datetime.datetime.now() - begin).total_seconds()
        print(elapsed)
        
        if regression:
            results_test["model"] = model_name_dict_reg[model_name]
        else:
            results_test["model"] = model_name_dict_class[model_name]
        results_test["set"] = "test"
        results_df.loc[len(results_df)] = results_test
        if regression:
            results_valid["model"] = model_name_dict_reg[model_name]
        else:
            results_valid["model"] = model_name_dict_class[model_name]
        results_valid["set"] = "valid"
        results_df.loc[len(results_df)] = results_valid

    results_df.to_csv(output_path)

