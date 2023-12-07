import pandas as pd
import numpy as np
from pathlib import Path
from auxiliary import validate_data



def load_data():

    data_dir    = Path("./data").resolve()
    cancer_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Cancer class dirs: {', '.join([str(d) for d in cancer_dirs])}")

    cancer_dict    = dict()

    # CANCER LEVEL
    for cancer_dir in cancer_dirs:
        print(f"\nLoading data from {cancer_dir}")
        case_dirs = [d for d in cancer_dir.iterdir() if d.is_dir()]
        print(f"Number of cases: {len(case_dirs)}")
        name_cancer_dir = cancer_dir.name
        
        # CASE LEVEL
        for case_dir in case_dirs:
            case_file = [f for f in case_dir.iterdir() if f.is_file() and f.suffix == ".tsv"]
            assert len(case_file) == 1
            case_file = case_file[0]
            
            case_data = pd.read_csv(case_file, sep="\t", skiprows = 1)
            case_data = case_data.drop(labels = case_data.columns[[0,3,4,5,7,8]], axis="columns")

            #entferne NA Zeilen und dann alle anderen Gene Type Zeilen
            case_data = case_data[case_data['gene_type'].notna()]
            case_data = case_data[case_data['gene_type'].str.contains('protein_coding')]
            case_data = case_data.drop(labels = case_data.columns[[1]], axis="columns")

            # Füge Krebstyp und Case Feature hinzu 
            case_data = case_data.T
            case_data = case_data.reset_index()
            case_data = case_data.drop(labels = case_data.columns[[0]], axis="columns")
            case_data.columns = case_data.iloc[0]
            case_data.insert(0, "Cancer", cancer_dir.name)
            case_data.insert(0, "Case", case_dir.name)
            case_data = case_data.drop(index = 0)
            case_dict  = {case_dir.name: np.delete(case_data.values, 0)}

            cancer_dict.update(case_dict)

    print(f"Length of Cancer Dictionary: {len(cancer_dict)}. Dataset Ready. \n")
    return cancer_dict
            



if __name__ == "__main__":
    #validate_data()
    load_data()