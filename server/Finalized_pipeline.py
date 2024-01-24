import pandas as pd
import numpy as np
import os
import datetime
from itertools import product
import pickle
import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt

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

import shap


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
def CalculateDescriptors(mol, drop_low_std=True):
    mol = mol.apply(Chem.MolFromSmiles)
    calc = Calculator(descriptors, ignore_3D=False)
    X_mordred = calc.pandas(mol, nproc=1)
    X_mordred = X_mordred.select_dtypes(['number'])
    #normalize
    X_mordred = (X_mordred-X_mordred.min())/(X_mordred.max()-X_mordred.min())
    #drop columns wth low std
    if drop_low_std:
        X_mordred = X_mordred.loc[:,X_mordred.std()>0.01]
    return X_mordred


def generate_split_dataset(path, train_fraction_split=0.8, output_csv_path=""):
    df = pd.read_csv(path)   

    ### Generate Split column
    train, test = train_test_split(df, train_size=train_fraction_split, random_state=42)
    train.insert(0, 'Split', 'train')
    test.insert(0, 'Split', 'test')
    df = pd.concat([train, test], axis=0)
    
    if output_csv_path != "":
        df.to_csv(output_csv_path)
    return df


def calculate_pIC50(input_df, IC50_column_name, output_csv_path=""): ### takes df or path as input
    if isinstance(input_df, str):
        path = input_df
        df = pd.read_csv(path)
    else:
        df = input_df

    if not (IC50_column_name in df.columns):
        print("Column does not exist")
        return
    
    df['pIC50'] = [-np.log10(i * 10**(-9)) for i in list(df[IC50_column_name])]
    
    if output_csv_path != "":
        df.to_csv(output_csv_path)
    return df


def calculate_classification_labels(input_df, pIC50_column_name, threshold=7, output_csv_path=""): ### takes df or path as input
    if isinstance(input_df, str):
        path = input_df
        df = pd.read_csv(path)
    else:
        df = input_df
    
    if not (pIC50_column_name in df.columns):
        print("Column does not exist")
        return
    
    df['label'] = [int(i > threshold) for i in list(df[pIC50_column_name])]
    
    if output_csv_path != "":
        df.to_csv(output_csv_path)
    return df


def calculate_features(input_df, calculate_descriptors, calculate_fingerprints, SMILES_column_name="SMILES", target_column_name="target", split_column_name="Split", output_csv_path=""): ### takes df or path as input
    if isinstance(input_df, str):
        path = input_df
        df = pd.read_csv(path)
    else:
        df = input_df

    if not (target_column_name in df.columns and SMILES_column_name in df.columns and split_column_name in df.columns):
        print("Column does not exist")
        return
    
    output_df = df[[target_column_name, split_column_name]]
    
    output_df.rename(columns={target_column_name: "Target", split_column_name: "Split"}, inplace=True)
    
    
    if calculate_descriptors:
        new_df = CalculateDescriptors(df[SMILES_column_name])
        output_df = pd.concat([output_df, new_df], axis=1)

    if calculate_fingerprints:
        new_df = CalculateMorganFingerprint(df[SMILES_column_name])
        output_df = pd.concat([output_df, new_df], axis=1)
    
    if output_csv_path != "":
        output_df.to_csv(output_csv_path)
    return output_df


def model_builder(model_name, hyperparams, regression):
    if model_name == 'dt':
        if regression:
            print(hyperparams)
            model = tree.DecisionTreeRegressor(**hyperparams)
        else:
            print(hyperparams)
            model = tree.DecisionTreeClassifier(**hyperparams)
    if model_name == 'rf':
        if regression:
            print(hyperparams)
            model = RandomForestRegressor(**hyperparams)
        else:
            print(hyperparams)
            model = RandomForestClassifier(**hyperparams)
            
    if model_name == 'lr':
        if regression:
            print(hyperparams)
            model = LinearRegression(**hyperparams)
        else:
            print(hyperparams)
            model = LogisticRegression(**hyperparams)

    if model_name == 'nn':
        if regression:
            print(hyperparams)
            model = MLPRegressor(**hyperparams)
        else:
            print(hyperparams)
            model = MLPClassifier(**hyperparams)
        
    if model_name == 'gb':
        if regression:
            print(hyperparams)
            model = GradientBoostingRegressor(**hyperparams)
        else:
            print(hyperparams)
            model = GradientBoostingClassifier(**hyperparams)

    if model_name == 'xg':
        if regression:
            print(hyperparams)
            model = XGBRegressor(**hyperparams)
        else:
            print(hyperparams)
            model = XGBClassifier(**hyperparams)
            
    if model_name == 'sv':
        if regression:
            print(hyperparams)
            model = SVR(**hyperparams)
        else:
            print(hyperparams)
            model = SVC(**hyperparams)
            
    return model


def train_and_test(model, X_train, y_train, X_test, y_test, regression, metrics=[], iterations=1):
    for i in range(iterations):
        model.fit(X_train, np.reshape(y_train, (-1, )))
        
        y_test_predicted = model.predict(X_test)

        y_train_predicted = model.predict(X_train)

        results_test = {}

        if regression:
            if 'rmse' in metrics or len(metrics) == 0:
                metric_test = mean_squared_error(y_test, y_test_predicted, squared=False)
                results_test["rmse"] = metric_test
                metric_test = mean_squared_error(y_train, y_train_predicted, squared=False)
                results_test["train_rmse"] = metric_test
            if 'mse' in metrics or len(metrics) == 0:
                metric_test = mean_squared_error(y_test, y_test_predicted)
                results_test["mse"] = metric_test
                metric_test = mean_squared_error(y_train, y_train_predicted)
                results_test["train_mse"] = metric_test
            if 'mae' in metrics or len(metrics) == 0:
                metric_test = mean_absolute_error(y_test, y_test_predicted)
                results_test["mae"] = metric_test
                metric_test = mean_absolute_error(y_train, y_train_predicted)
                results_test["train_mae"] = metric_test
            if 'r2' in metrics or len(metrics) == 0:
                metric_test = r2_score(y_test, y_test_predicted)
                results_test["r2"] = metric_test
                metric_test = r2_score(y_train, y_train_predicted)
                results_test["train_r2"] = metric_test
            
        else:
            if 'roc_auc' in metrics or len(metrics) == 0:
                metric_test = roc_auc_score(y_test, y_test_predicted)
                results_test["roc_auc"] = metric_test
                metric_test = roc_auc_score(y_train, y_train_predicted)
                results_test["train_roc_auc"] = metric_test
            if 'accuracy' in metrics or len(metrics) == 0:
                metric_test = accuracy_score(y_test, y_test_predicted)
                results_test["accuracy"] = metric_test
                metric_test = accuracy_score(y_train, y_train_predicted)
                results_test["train_accuracy"] = metric_test
            if 'precision' in metrics or len(metrics) == 0:
                metric_test = precision_score(y_test, y_test_predicted)
                results_test["precision"] = metric_test
                metric_test = precision_score(y_train, y_train_predicted)
                results_test["train_precision"] = metric_test
            if 'recall' in metrics or len(metrics) == 0:
                metric_test = recall_score(y_test, y_test_predicted)
                results_test["recall"] = metric_test
                metric_test = recall_score(y_train, y_train_predicted)
                results_test["train_recall"] = metric_test
            if 'f1' in metrics or len(metrics) == 0:
                metric_test = f1_score(y_test, y_test_predicted)
                results_test["f1"] = metric_test
                metric_test = f1_score(y_train, y_train_predicted)
                results_test["train_f1"] = metric_test

    return results_test


def split_df(df):
    
    train = df[df['Split'] == 'train']
    test = df[df['Split'] == 'test']

    X_train = train.drop(['Target', 'Split'], axis=1)
    X_train = X_train.drop([list(X_train.columns)[0]], axis=1)
    y_train = train[['Target']]
    X_test = test.drop(['Target', 'Split'], axis=1)
    X_test = X_test.drop([list(X_test.columns)[0]], axis=1)
    y_test = test[['Target']]

    return X_train, y_train, X_test, y_test

### input df can be a df or a csv to read
def hyperparameter_search(input_df, parameters, unique=True, output_file_name="results.csv"):
    model_name_dict_reg = {"dt": "DecisionTreeRegressor", "rf": "RandomForestRegressor", "lr": "LinearRegression", "nn": "MLPRegressor", "gb": "GradientBoostingRegressor", "xg": "XGBRegressor", "sv": "SVR"}
    model_name_dict_class = {"dt": "DecisionTreeClassifier", "rf": "RandomForestClassifier", "lr": "LogisticRegression", "nn": "MLPClassifier", "gb": "GradientBoostingClassifier", "xg": "XGBClassifier", "sv": "SVC"}
    

    if isinstance(input_df, str):
        path = input_df
        df = pd.read_csv(path)
    else:
        df = input_df

    if isinstance(parameters, dict):
        parameters = list(parameters.items())

    models_with_parameters = []

    for model_name, parameter_grid in parameters:
        keys = parameter_grid.keys()
        param_combinations = list(product(*parameter_grid.values()))
        for combination in param_combinations:
            param_set = {}
            for index, k in enumerate(keys):
                param_set[k] = combination[index]
            models_with_parameters.append([model_name, param_set])

    
    metrics = []

    if df['Target'].nunique() > 2:
        regression=True
    else:
        regression=False

    print("Regression: ", regression)

    data = {"model": [], "hyperparams": []}
    if regression:
        for metric in ["mse", "rmse", "mae", "r2"]:
            data[metric] = []
            data["train_" + metric] = []
    else:
        for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            data[metric] = []
            data["train_" + metric] = []

    best_model = None
    if regression:
        compared_score = "mse"
        best_model_score = 100000
    else:
        compared_score = "roc_auc"
        best_model_score = 0


    output_path = "experiments\Standardized Pipeline\\" + output_file_name
    if unique:
        output_path = uniquify(output_path)
    X_train, y_train, X_test, y_test = split_df(df)

    results_df = pd.DataFrame(data)

    i = -1
    for model_name, hyperparams in models_with_parameters:
        i += 1
        print(model_name)

        model = model_builder(model_name, hyperparams, regression)
        
        begin = datetime.datetime.now()
        
        results_test = train_and_test(model, X_train, y_train, X_test, y_test, regression=regression, metrics=metrics)
        
        elapsed = (datetime.datetime.now() - begin).total_seconds()
        print("time: ", elapsed)

        
        if regression:
            results_test["model"] = model_name_dict_reg[model_name]
        else:
            results_test["model"] = model_name_dict_class[model_name]
        results_test["hyperparams"] = str(hyperparams)
        
        results_df.loc[len(results_df)] = results_test


        if regression and results_test[compared_score] < best_model_score:
            best_model_score = results_test[compared_score]
            best_model = model
        if not regression and results_test[compared_score] > best_model_score:
            best_model_score = results_test[compared_score]
            best_model = model

    results_df = results_df.sort_values(by=[compared_score], ascending=False)

    return results_df, best_model

    
def retrain_model(model, input_df):
    ### Read model
    if isinstance(model, str):
        model_path = model
        model = pickle.load(open(model_path, 'rb'))

    
    ### prepare dataset (shold be split and features calculated already)
    if isinstance(input_df, str):
        path = input_df
        df = pd.read_csv(path)
    else:
        df = input_df

    if df['Target'].nunique() > 2:
        regression=True
    else:
        regression=False

    X_train, y_train, X_test, y_test = split_df(df)

    ### train
    results_test = train_and_test(model, X_train, y_train, X_test, y_test, regression=regression, metrics=[])
        

    ### print/export new results and new model
    output_path = "experiments\Standardized Pipeline\\"
    
    filename = os.path.join(os.path.dirname(output_path), 'retrained_model.sav')
    pickle.dump(model, open(filename, 'wb'))

    return results_test

def make_prediction(model, input_SMILES, calculate_descriptors, calculate_fingerprints, SMILES_column_name='mol'):
    
    ### Read model
    if isinstance(model, str):
        model_path = model
        model = pickle.load(open(model_path, 'rb'))
    
    
    ### Check if only one smiles, and if it needs to be put into a df

    input_df = pd.DataFrame()
    
    ### calculate features
    if calculate_descriptors:
        new_df = CalculateDescriptors(input_SMILES[SMILES_column_name], drop_low_std=False)
        input_df = pd.concat([input_df, new_df], axis=1)

    if calculate_fingerprints:
        new_df = CalculateMorganFingerprint(input_SMILES[SMILES_column_name])
        input_df = pd.concat([input_df, new_df], axis=1)

    input_df = input_df.filter(model.feature_names_in_, axis=1)
    input_df = input_df.fillna(0)
    
    ### make prediction on these features
    predicted = model.predict(X=input_df)

    tree_models = [tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, RandomForestClassifier, RandomForestRegressor, XGBClassifier, XGBRegressor, GradientBoostingClassifier, GradientBoostingRegressor]
    kernel_models = [SVC, SVR, MLPClassifier, MLPRegressor, LinearRegression, LogisticRegression]
    if type(model) in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        plot = shap.summary_plot(shap_values, input_df, show=False)
    if type(model) in kernel_models:
        explainer = shap.KernelExplainer(model)
        shap_values = explainer.shap_values(input_df)
        # plot = shap.summary_plot(shap_values, input_df, show=False)
    ## return the label/pIC50 value
    # plt.savefig('explainability_plot.png')
    return predicted
