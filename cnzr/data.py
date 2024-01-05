from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Union

CANCER_TYPES = {
    # Übersetzung
    "tcga-kich": 0,
    "tcga-kirc": 1,
    "tcga-kirp": 2,
}

def get_cancer_type_by_id(id: int):
    for k, v in CANCER_TYPES.items():
        if v == id:
            return k
    return None

def load_data(root: Union[str, Path]):
    data_dir = root if isinstance(root, Path) else Path(root)
    data_dir = data_dir.resolve()
    preprocessed_file = data_dir / "preprocessed.pkl"
    if preprocessed_file.exists():
        print("Loading preprocessed data...")
        with open(preprocessed_file, "rb") as f:
            x, y = pickle.load(f)
            return x, y
    data = []
    cancer_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Cancer class dirs: {', '.join([str(d) for d in cancer_dirs])}")

    # CANCER LEVEL
    for cancer_dir in cancer_dirs:
        print(f"\nLoading data from {cancer_dir}")
        case_dirs = [d for d in cancer_dir.iterdir() if d.is_dir()]
        print(f"Number of cases: {len(case_dirs)}")
        # CASE LEVEL
        for case_dir in case_dirs:
            case_file = [f for f in case_dir.iterdir() if f.is_file() and f.suffix == ".tsv"]
            assert len(case_file) == 1
            case_data = pd.read_csv(case_file[0], sep="\t", skiprows = 1)
            case_data = case_data.drop(labels = case_data.columns[[0,3,4,5,7,8]], axis="columns")

            #entferne NA Zeilen und dann alle anderen Gene Type Zeilen
            case_data = case_data[case_data['gene_type'].notna()]
            case_data = case_data[case_data['gene_type'].str.contains('protein_coding')]
            case_data = case_data.drop(labels = case_data.columns[[1]], axis="columns")
            x = case_data["tpm_unstranded"].to_numpy()
            y = np.full(x.shape, cancer_dir.name)
            data.append({
                "id": case_dir.name,
                "x": x,
                "y": y
            })

    data = list(sorted(data, key=lambda x: x["id"]))
    n_cases = len(data)
    n_genes = data[0]["x"].shape[0]
    x = np.zeros((n_cases, n_genes))
    y = np.zeros(n_cases)

    for i, case_data in enumerate(data):
        x[i] = case_data["x"]
        y[i] = CANCER_TYPES[case_data["y"].astype(str)[0]]

    # save dict
    with open(data_dir / "preprocessed.pkl", "wb") as f:
        pickle.dump((x, y), f)
    return x, y


def one_hot_encode(y: np.ndarray, n_classes: int):
    y_t = np.zeros((len(y), n_classes))
    for i, v in enumerate(y):
        y_t[i][int(v)] = 1
    return y_t
