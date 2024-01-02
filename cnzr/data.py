
from auxiliary import validate_data
from training import ModelV1, train_model, test_model
from sklearn.utils import shuffle
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import time



CANCER_TYPES = {
    # Übersetzung
    "tcga-kich": 0,
    "tcga-kirc": 1,
    "tcga-kirp": 2,
}



def load_data():
    data_dir    = Path("./data").resolve()
    preprocessed_file = data_dir / "preprocessed.pkl"
    if preprocessed_file.exists():
        print("Loading preprocessed data...")
        with open(preprocessed_file, "rb") as f:
            x, y = pickle.load(f)
            return x, y
    cancer_dict = dict()
    cancer_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Cancer class dirs: {', '.join([str(d) for d in cancer_dirs])}")


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
            case_data = pd.read_csv(case_file[0], sep="\t", skiprows = 1)
            case_data = case_data.drop(labels = case_data.columns[[0,3,4,5,7,8]], axis="columns")

            #entferne NA Zeilen und dann alle anderen Gene Type Zeilen
            case_data = case_data[case_data['gene_type'].notna()]
            case_data = case_data[case_data['gene_type'].str.contains('protein_coding')]
            case_data = case_data.drop(labels = case_data.columns[[1]], axis="columns")
            x = case_data["tpm_unstranded"].to_numpy()
            y = np.full(x.shape, cancer_dir.name)
            cancer_dict[case_dir.name] = {
                "x": x,
                "y": y
            }
    n_cases = len(cancer_dict)
    n_genes = cancer_dict[next(iter(cancer_dict))]["x"].shape[0]
    x = np.zeros((n_cases, n_genes))
    y = np.zeros(n_cases)

    for i, (case_name, case_data) in enumerate(cancer_dict.items()):
        x[i] = case_data["x"]
        y[i] = CANCER_TYPES[case_data["y"].astype(str)[0]]
    print(f"Length of Cancer Dictionary: {len(cancer_dict)}. Dataset Ready. \n")
    # save dict
    with open(data_dir / "preprocessed.pkl", "wb") as f:
        pickle.dump((x, y), f)
    return x, y



if __name__ == "__main__":

    ratio_train_test = 0.7

    # validate_data()
    x, y = load_data()

    ## Bring y into the right form ([y1 = 1, y2 = 0, y3 =0] z.B.) 
    y_t = np.zeros((len(y), len(CANCER_TYPES)))
    for idx, value in enumerate(y):
        y_t[idx][int(value)] = 1
    y = y_t
    del y_t

    #Train Test Split
    random_seed = 42
    x, y    = shuffle(x, y, random_state=0)
    idx     = int(ratio_train_test*len(x))
    x_train = x[range(1, idx)]
    y_train = y[range(1, idx)]
    x_test  = x[range(idx+1, len(x))]
    y_test  = y[range(idx+1, len(y))]
    
    # Model
    start_time = time.time()
    NN          = ModelV1(np.shape(x)[1], len(CANCER_TYPES) )
    Classifier  = train_model(x_train, y_train, NN, n_classes= len(CANCER_TYPES))
    test_model(x_test, y_test, Classifier)
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time)/ 60} minutes")