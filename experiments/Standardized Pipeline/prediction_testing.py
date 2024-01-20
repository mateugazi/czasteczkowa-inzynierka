import os
import pandas as pd
import Finalized_pipeline


datasets = []
for (dirpath, dirnames, filenames) in os.walk(r"experiments\split_datasets"):
    datasets.extend(filenames)
    break

chosen_dataset = os.path.join(r"experiments\split_datasets", datasets[0])
chosen_df = pd.read_csv(chosen_dataset)
chosen_entries = chosen_df.iloc[1:300]
print(chosen_dataset)

#print(chosen_entry)

if 'SMILES' in chosen_df.columns:
    smiles_df = chosen_entries['SMILES']
elif 'mol' in chosen_df.columns:
    smiles_df = chosen_entries['mol']
else:
    print("MISSING SMILES")


#print(smiles_df)

model_path = r"experiments\Standardized Pipeline\model.sav"

pred = Finalized_pipeline.make_prediction(model_path, smiles_df, calculate_descriptors=True, calculate_fingerprints=False)
print(pred)