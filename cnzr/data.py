import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    data_dir = Path("./data").resolve()
    cancer_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Cancer class dirs: {', '.join([str(d) for d in cancer_dirs])}")

    cancer_type_data = dict()

    for cancer_dir in cancer_dirs:
        print(f"Loading data from {cancer_dir}")
        case_dirs = [d for d in cancer_dir.iterdir() if d.is_dir()]
        print(f"Number of cases: {len(case_dirs)}")

        for case_dir in case_dirs:
            case_file = [f for f in case_dir.iterdir() if f.is_file() and f.suffix == ".tsv"]
            assert len(case_file) == 1
            case_file = case_file[0]

            data = pd.read_csv(case_file, sep="\t", header=None)


if __name__ == "__main__":
    load_data()