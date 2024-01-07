# %%
import warnings
from itertools import product
import numpy as np
import pandas as pd
#from mordred import Calculator, descriptors
#from rdkit import Chem
#from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import (StratifiedKFold, KFold,
                                     cross_val_score, train_test_split)

from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=Warning)

N_SPLITS = 2
RANDOM_STATE = 148260

# %% [markdown]
# ### Preprocessing

# %%
def LoadCSV(path):
    df = pd.read_csv(path)
    return df

def LoadCSV_BACE(path, regression = False):
    df = pd.read_csv(path)
    df.drop_duplicates('mol')
    df = df.dropna()
    df.drop(['CID', 'canvasUID'], axis=1, inplace=True)
    if regression:
        df['Target'] = df['pIC50']
        df.drop('Class', axis=1, inplace=True)
        df.drop('pIC50', axis=1, inplace=True)
    else:
        df['Target'] = df['Class']
        df.drop('Class', axis=1, inplace=True)
        df.drop('pIC50', axis=1, inplace=True)
    return df

from rdkit import Chem
from rdkit.Chem import AllChem
def CalculateMorganFingerprint(mol):
    mfpgen = AllChem.GetMorganGenerator(radius=2,fpSize=1024)
    fingerprint = np.array([mfpgen.GetFingerprintAsNumPy(x) for x in mol])
    fingerprint = pd.DataFrame(fingerprint, columns = ['mfp'+str(i) for i in range(fingerprint.shape[1])])
    return fingerprint

def split_data_BACE(df, scaffold=True):
    X = df.drop(['Target'], axis=1)

    X['mol_conv'] = X['mol'].apply(Chem.MolFromSmiles)

    X = CalculateMorganFingerprint(X['mol_conv'])

    X['Model'] = df['Model']

    #X = X.drop(['mol'], axis=1)

    if not scaffold:
        y = df[['Target']]
        X = X.drop(['Model'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.111, random_state=42)
        return X_train, y_train, X_test, y_test, X_valid, y_valid

    #dummy = list(X['Model'])
    #
    #sc = StandardScaler()
    #X = sc.fit_transform(X.drop(['Model'], axis=1))
    #X = pd.DataFrame(X)
    #
    #X['Model'] = dummy

    X_train = X[X['Model'] == 'Train']
    X_test = X[X['Model'] == 'Test']
    X_valid = X[X['Model'] == 'Valid']

    y = df[['Target', 'Model']]

    y_train = y[y['Model'] == 'Train']
    y_test = y[y['Model'] == 'Test']
    y_valid = y[y['Model'] == 'Valid']
    
    X_train.drop('Model', axis=1, inplace=True)
    X_test.drop('Model', axis=1, inplace=True)
    X_valid.drop('Model', axis=1, inplace=True)
    y_train.drop('Model', axis=1, inplace=True)
    y_test.drop('Model', axis=1, inplace=True)
    y_valid.drop('Model', axis=1, inplace=True)
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid

# %%
df_regression = LoadCSV_BACE(r"C:\Users\Wojciech\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv", regression=True)
df_classification = LoadCSV_BACE(r"C:\Users\Wojciech\Documents\GitHub\czasteczkowa-inzynierka\experiments\BACE\bace.csv")

# %%
scaffold = True

# %%
X_train_class, y_train_class, X_test_class, y_test_class, X_valid_class, y_valid_class = split_data_BACE(df_classification, scaffold=scaffold)

# %%
X_train_regre, y_train_regre, X_test_regre, y_test_regre, X_valid_regre, y_valid_regre = split_data_BACE(df_regression, scaffold=scaffold)

# %%
df_classification.head()

# %%
y_train_regre

# %%
print(X_train_class.shape)
print(X_test_class.shape)
print(X_valid_class.shape)
print(f"{round(X_train_class.shape[0] / df_classification.shape[0], 2)}")
print(f"{round(X_test_class.shape[0] / df_classification.shape[0], 2)}")
print(f"{round(X_valid_class.shape[0] / df_classification.shape[0], 2)}")

# %%
df_classification.describe()

# %% [markdown]
# ### Run configurations

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
def results_metrics(y_true, y_pred, regression=False):
    if regression:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return [mae, rmse, mse, r2]
    else:
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [accuracy, roc_auc, precision, recall, f1]

# %%
def run_rf(X_train, X_test, X_valid, y_train, y_test, y_valid, n_estimators, max_depth, min_samples_split, min_samples_leaf, regression=False):
    
    if regression:
        name = "RandomForestRegressor"
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    else:
        name = "RandomForestClassifier"
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    results = results_metrics(y_test, y_predicted, regression)

    output_str = f"{name}-{n_estimators}-{max_depth}-{min_samples_split}-{min_samples_leaf}; "
    if regression: output_str += f"MAE: {results[0]} | RMSE: {results[1]} | MSE: {results[2]} | R2: {results[3]}"
    else: output_str += f"Accuracy: {results[0]} | ROC-AUC: {results[1]} | Precision: {results[2]} | Recall: {results[3]} | F1: {results[4]}"
    return (output_str)

def run_lr(X_train, X_test, X_valid, y_train, y_test, y_valid, C, penalty, solver, regression=False):
    if regression:
        name = "LinearRegression"
        model = LinearRegression()
    else:
        name = "LogisticRegression"
        model = LogisticRegression(C=C, penalty=penalty, solver=solver)
        
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    results = results_metrics(y_test, y_predicted, regression)

    output_str = f"{name}-{C}-{penalty}-{solver}; "
    if regression: output_str += f"MAE: {results[0]} | RMSE: {results[1]} | MSE: {results[2]} | R2: {results[3]}"
    else: output_str += f"Accuracy: {results[0]} | ROC-AUC: {results[1]} | Precision: {results[2]} | Recall: {results[3]} | F1: {results[4]}"
    return (output_str)

def run_nn(X_train, X_test, X_valid, y_train, y_test, y_valid, hidden_layer_sizes, activation, alpha, max_iter, regression=False):
    if regression:
        name = "MLPRegressor"
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, max_iter=max_iter)
    else:
        name = "MLPClassifier"
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, max_iter=max_iter)
         
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    results = results_metrics(y_test, y_predicted, regression)

    output_str = f"{name}-{hidden_layer_sizes}-{activation}-{alpha}-{max_iter}; "
    if regression: output_str += f"MAE: {results[0]} | RMSE: {results[1]} | MSE: {results[2]} | R2: {results[3]}"
    else: output_str += f"Accuracy: {results[0]} | ROC-AUC: {results[1]} | Precision: {results[2]} | Recall: {results[3]} | F1: {results[4]}"
    return (output_str)

def run_gb(X_train, X_test, X_valid, y_train, y_test, y_valid, n_estimators, learning_rate, regression=False):
    if regression:
        name = "GradientBoostingRegressor"
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    else:
        name = "GradientBoostingClassifier"
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    results = results_metrics(y_test, y_predicted, regression)
    
    output_str = f"{name}-{n_estimators}-{learning_rate}; "
    if regression: output_str += f"MAE: {results[0]} | RMSE: {results[1]} | MSE: {results[2]} | R2: {results[3]}"
    else: output_str += f"Accuracy: {results[0]} | ROC-AUC: {results[1]} | Precision: {results[2]} | Recall: {results[3]} | F1: {results[4]}"
    return output_str

def run_svm(X_train, X_test, X_valid, y_train, y_test, y_valid, c, d, e, regression=False):
    if regression:
        name = "SVR"
        model = SVR(C=c, degree=d, epsilon=e, kernel="poly")
    else:
        name = "SVC"
        model = SVC(C=c, degree=d, kernel="poly") ### Epsilon is ignored

    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    results = results_metrics(y_test, y_predicted, regression)
    
    output_str = f"{name}-{c}-{d}-{e}; "
    if regression: output_str += f"MAE: {results[0]} | RMSE: {results[1]} | MSE: {results[2]} | R2: {results[3]}"
    else: output_str += f"Accuracy: {results[0]} | ROC-AUC: {results[1]} | Precision: {results[2]} | Recall: {results[3]} | F1: {results[4]}"
    return output_str

# %%
def run_rf_specific(X_train, X_test, X_valid, y_train, y_test, y_valid, regression=False):
    if regression:
        bootstrap = True
        criterion = "squared_error"
        min_samples_split = 32
        n_estimators = 100
        name = "RandomForestRegressor"
        model = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, criterion=criterion, bootstrap=bootstrap)
    else:
        bootstrap = True
        criterion = "entropy"
        min_samples_split = 32
        n_estimators = 30
        name = "RandomForestClassifier"
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, criterion=criterion, bootstrap=bootstrap)
        
    
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    results = results_metrics(y_test, y_predicted, regression)

    output_str = f"{name}-SPECIFIC RUN; "
    if regression: output_str += f"MAE: {results[0]} | RMSE: {results[1]} | MSE: {results[2]} | R2: {results[3]}"
    else: output_str += f"Accuracy: {results[0]} | ROC-AUC: {results[1]} | Precision: {results[2]} | Recall: {results[3]} | F1: {results[4]}"
    return (output_str)
    

# %%
def run_all(X_train, X_test, X_valid, y_train, y_test, y_valid, regression=False):
    results = []

    print("Run")

    #### -----

    param_grid_rf={
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_combinations = list(product(*param_grid_rf.values()))
    for combination in param_combinations:
        n, m, s, l = combination
        results.append(run_rf(X_train, X_test, X_valid, y_train, y_test, y_valid, n, m, s, l, regression))
        print(results[-1])
    ### -----

    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    if regression:
        param_grid_lr = {
            'C': [0.001],
            'penalty': ['l1'],
            'solver': ['liblinear']
        }
    param_combinations = list(product(*param_grid_lr.values()))
    for combination in param_combinations:
        C, p, s = combination
        results.append(run_lr(X_train, X_test, X_valid, y_train, y_test, y_valid, C, p, s, regression))
        print(results[-1])
    ### -----

    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500, 1000]
    }
    param_combinations = list(product(*param_grid_mlp.values()))
    for combination in param_combinations:
        h, ac, a, i = combination
        results.append(run_nn(X_train, X_test, X_valid, y_train, y_test, y_valid, h, ac, a, i, regression))
        print(results[-1])
    ### -----

    param_grid_gb={
        'n_estimators': [10, 100, 200], 
        'learning_rate': [0.1,0.5,1.0,2.0]
    }
    param_combinations = list(product(*param_grid_gb.values()))
    for combination in param_combinations:
        n, lr = combination
        results.append(run_gb(X_train, X_test, X_valid, y_train, y_test, y_valid, n, lr, regression))
        print(results[-1])
    ### -----
    
    param_grid_svm = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'degree': [2, 3, 4, 5],
        'epsilon': ["no epsilon"]
    }
    
    if regression:
        param_grid_svm = {
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'degree': [2, 3, 4],
            'epsilon': [0.01, 0.1, 1]
        }
        param_combinations = list(product(*param_grid_svm.values()))
        for combination in param_combinations:
            c, d, e = combination
            results.append(run_svm(X_train, X_test, X_valid, y_train, y_test, y_valid, c, d, e, regression))
            print(results[-1])
    
    return results

# %%
def run_configured(regression=False, pca=False, specific_run=False):
    sc = StandardScaler()

    if regression:
        X_train = X_train_regre
        X_test =  X_test_regre
        X_valid = X_valid_regre
        y_train = y_train_regre
        y_test =  y_test_regre
        y_valid = y_valid_regre
    else:
        X_train = X_train_class
        X_test =  X_test_class
        X_valid = X_valid_class
        y_train = y_train_class
        y_test =  y_test_class
        y_valid = y_valid_class

    if pca:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test =  pca.fit_transform(X_test)
        X_valid = pca.fit_transform(X_valid)

    if specific_run:
        if pca: print("With PCA:")
        else: print("Without PCA:")
        results = run_rf_specific(X_train, X_test, X_valid, y_train, y_test, y_valid, regression=regression)
        csv_path = "BACE_specific_run_"
        if regression: csv_path += "regression" 
        else: csv_path += "classification"
        if pca: csv_path += "_pca"
        print(results)
        #data_tuples = [tuple(item.split('; ')) for item in results]
        #df = pd.DataFrame(data_tuples, columns=['Classifier', 'Accuracy'])
        #df.to_csv(csv_path)
        return

    results = run_all(X_train, X_test, X_valid, y_train, y_test, y_valid, regression=regression)

    csv_path = "BACE_comparison_results_"
    if scaffold: csv_path = "Scaffold/" + csv_path
    else: csv_path = "No_scaffold/" + csv_path
    if regression: csv_path += "regression" 
    else: csv_path += "classification"
    if not scaffold: csv_path += "_no"
    csv_path += "_scaffold"
    if pca: csv_path += "_pca"
    csv_path += "_svm.csv"

    data_tuples = [tuple(item.split('; ')) for item in results]
    df = pd.DataFrame(data_tuples, columns=['Classifier', 'Accuracy'])
    df.to_csv(csv_path)

    return

# %%
run_grid = {
        'regression': [False, True],
        'pca': [False]
    }

run_param_combinations = list(product(*run_grid.values()))

for combination in run_param_combinations:
    r, p = combination
    run_configured(r, p, specific_run=True) ### This arguments runs only rf with a specific configuration


