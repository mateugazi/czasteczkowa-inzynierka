import os
import pandas as pd
import Finalized_pipeline

calculate_descriptors = False
calculate_fingerprints = True

for regression in [True]:
    if regression:
        param_grid_dt={}
    else:
        param_grid_dt={}
    
    datasets = []
    for (dirpath, dirnames, filenames) in os.walk(r"experiments\split_datasets"):
        datasets.extend(filenames)
        break


    for dataset in datasets[6:7]:
        ### Check which files will be used:
        #print(dataset)
        #continue

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
                                                        split_column_name="Split")
        #print(df.columns)        
        hyperparams = {"dt": param_grid_dt}
    
        Finalized_pipeline.hyperparameter_search(df, hyperparams, output_file_name="test_run.csv", unique=False)


        ###
        ### Retraining part
        ###


        model_path = r"experiments\Standardized Pipeline\model.sav"
        retrained_dataset_path = os.path.join(r"experiments\split_datasets", datasets[8])
        
        df_retrain = pd.read_csv(retrained_dataset_path)
        if not "pIC50" in df_retrain.columns:
            df_retrain = Finalized_pipeline.calculate_pIC50(df_retrain, "target")
        if not regression:
            df_retrain = Finalized_pipeline.calculate_classification_labels(df_retrain, "pIC50", threshold=7)
            target_column = "label"
        else:
            target_column = "pIC50"

        if "mol" in df_retrain.columns:
            df_retrain.rename(columns={"mol": "SMILES"}, inplace=True)
            
        if regression:
            runtype = "regression"
        else:
            runtype = "classification"
        df_retrain = Finalized_pipeline.calculate_features(df_retrain, calculate_descriptors=calculate_descriptors, calculate_fingerprints=calculate_fingerprints, 
                                                            SMILES_column_name="SMILES", target_column_name=target_column, 
                                                            split_column_name="Split")
            
        print(Finalized_pipeline.retrain_model(model_path, df_retrain))

        break
