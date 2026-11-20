import glob
import sys

import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

# Script for processing data from csv files in folder data/raw_data or custom path
# Creates train, test, and holdout validation datasets
# First split 80/20 train/test
# Second split 90/10 train/val for holdout validation
# Writes 8 csv files

# Constants
SAMPLE_SIZE = 6000
SAVE_NAMES = [
    'x_train_kfold.csv',
    'x_test.csv',
    'y_train_kfold.csv',
    'y_test.csv'
]
SAVE_NAMES_HOLDOUT = [
    'x_train_holdout.csv',
    'x_val_holdout.csv',
    'y_train_holdout.csv',
    'y_val_holdout.csv'
]
COLUMNS_TO_DROP = [
    # Constants
    'protocol', # constant = 6
    'bwd_psh_flags', # constant = 0
    'bwd_urg_flags', # 0
    'fwd_bytes/bulk_avg', # 0
    'fwd_packet/bulk_avg', # 0
    'fwd_bulk_rate_avg', # 0
    # Data Leak
    'label', # Boolean for malicious traffic
    # Strings
    'flow_id',
    'src_ip',
    'dst_ip',
    'timestamp'
]
TARGETS_TO_DROP = [
    'Recon Ping Sweep',
    'MITM ARP Spoofing'
]

# Strips leading and trailing whitespace, replaces " " with "_", sets to lowercase
def format_column_names(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.lower()
    )
    return dataframe

# Gets file path of dataset csv files
def get_data_path():
    if len(sys.argv) == 2:
        return Path(sys.argv[1])
    script_folder = Path(__file__).resolve().parent
    return script_folder.parent.parent / 'data' / 'raw_data'

# Loads csv files into dataframe
def load_csv_data(file_path):
    csv_files = list(file_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError("No csv files found in {}".format(file_path))
    df_list: list[pd.DataFrame] = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)

# Samples by attack_name, shuffling before and after selection
def sample_or_take_all(dataframe: pd.DataFrame, column_group: str, sample_size: int):
    return (
        dataframe
        .groupby(column_group, group_keys=False)
        .sample(frac=1, random_state=42)
        .groupby(column_group, group_keys=False)
        .head(sample_size)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

# Saves a dataframe to csv in data/cleaned_data
def save_to_csv(dataframe: pd.DataFrame, filename: str):
    script_folder = Path(__file__).resolve().parent
    save_folder = script_folder.parent.parent / 'data' / 'processed_data'
    save_folder.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(save_folder / filename, index=False)

# Performs test_train_split and saves resulting dataframes in csv files named by list of file names
def split_and_save(x: pd.DataFrame, y: pd.Series, save_names: list[str], test_size=0.2, random_state=42):
    split_dfs = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    assert len(split_dfs) == len(save_names)
    for df, save_name in zip(split_dfs, save_names):
        save_to_csv(df, save_name)
    return split_dfs

# Being processing dataframe
print('Processing data...', end=' ', flush=True)

# Formats column names, drops problem columns, samples by attack type
df = (
    load_csv_data(get_data_path())
    .pipe(format_column_names)
    .drop(columns=COLUMNS_TO_DROP)
    .dropna()
    .loc[lambda df: ~df['attack_name'].isin(TARGETS_TO_DROP)]
    .pipe(sample_or_take_all, 'attack_name', SAMPLE_SIZE)
)

# Separate features and targets
features = df.drop(columns=['attack_name'])
targets = df['attack_name']

# Split train and test data; then further split train data into holdout train and test data
first_split = split_and_save(features, targets, SAVE_NAMES)
second_split = split_and_save(first_split[0], first_split[2], SAVE_NAMES_HOLDOUT, test_size=0.1, random_state=42)

print('done')