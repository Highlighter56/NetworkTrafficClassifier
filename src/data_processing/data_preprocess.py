import glob
import sys

import pandas as pd

# Gets file path of dataset csv files
# Checks for custom path, path from src, path from root
def get_data_path():
    path_from_root = "data/raw_data/*.csv"
    if len(sys.argv) == 2:
        return sys.argv[1] + "*.csv"
    elif len(sys.argv) == 15:
        return "../" + path_from_root
    else:
        return path_from_root

# Loads csv files into dataframe
def load_csv_data(file_path):
    csv_files = glob.glob(file_path)
    df_list: list[pd.DataFrame] = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)

# Start here
df = load_csv_data(get_data_path())