import pandas as pd
import numpy as np
from pathlib import Path



def validate_data():
    # Überprüfe, ob in den Case-Dateien die Selbe Menge von Proteinen steckt.
    data_dir    = Path("./data").resolve()
    cancer_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Cancer class dirs: {', '.join([str(d) for d in cancer_dirs])}")

    found_protein_codes = np.zeros(len(cancer_dirs))           # zum Vergelich des Vorkommens Protein-codierender Gene in verschiedenen Daten
    i_cancer_dirs = 0
    for cancer_dir in cancer_dirs:
        case_dirs = [d for d in cancer_dir.iterdir() if d.is_dir()]
        gene_type_protein = np.zeros(len(case_dirs))
        i_case_dir = 0     
        for case_dir in case_dirs:
            case_file = [f for f in case_dir.iterdir() if f.is_file() and f.suffix == ".tsv"]
            assert len(case_file) == 1
            case_file = case_file[0]
            data = pd.read_csv(case_file, sep="\t", skiprows = 1)
            data = data.drop(labels = data.columns[[0,3,4,5,7,8]], axis="columns")
            # Zähle Zeilenanzahl von Protein-Coding Types
            gene_type_protein[i_case_dir] = data.value_counts("gene_type").iloc[0]
            # Debug            
            if i_case_dir == len(case_dirs)-1:
                unique, counts = np.unique(gene_type_protein, return_counts=True)
                print(dict(zip(unique, counts)))
                found_protein_codes[i_cancer_dirs] = unique[0]
            i_case_dir += 1         
        i_cancer_dirs += 1     
    print(f"*debug*\n found_protein_codes[0] = {found_protein_codes[0]} \n np.sum(found_protein_codes)= {np.sum(found_protein_codes)} \n len(found_protein_codes)= {len(found_protein_codes)}")
    print(f"ergibt: {found_protein_codes[0]} == {np.sum(found_protein_codes)/len(found_protein_codes)}")
    assert found_protein_codes[0] == np.sum(found_protein_codes)/len(found_protein_codes)
    print("Überprüfung erfolgreich.")

