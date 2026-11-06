# Python script for cleaning data files
import glob
import sys
import pandas as pandas
import numpy as numpy
from sklearn.model_selection import train_test_split


# 
def load_csv_data():
    path = ""

	# If Entered a Custom Path
    if len(sys.argv) == 2:
        return glob.glob(sys.argv[1] + "*.csv")

	# If Entered no Path
    # If running from scr
    file_pathsA = glob.glob("../data/source_data/*.csv")
    if len(file_pathsA) == 15:
          return file_pathsA
    # If running from root
    return glob.glob("data/source_data/*.csv")

	

def sample_or_take_all(group):
    sample_size = min(len(group), 2000)
    return group.sample(n=sample_size, random_state=42)

def data_cleaning():
    # Store the loose csv files into list

    file_paths = load_csv_data()

	# Testing you can run from src, root, or a custome destination provided you specify where the data is
    # print("File Path Size:",len(file_paths))
    # sys.exit(1)

    # Build a dataframe for each csv file and store in list
    dataframe_list = [pandas.read_csv(file) for file in file_paths]

    # Concatenate each dataframe in list and reset index
    dataframe_combined = pandas.concat(dataframe_list, ignore_index=True)
    dataframe_combined = dataframe_combined.reset_index(drop=True)

    # Drop rows will null values
    dataframe_combined.dropna()

    # Drop problematic Attack Names
    dataframe_filtered_rps = dataframe_combined[dataframe_combined['Attack Name'] != 'Recon Ping Sweep']
    dataframe_filtered = dataframe_filtered_rps[dataframe_filtered_rps['Attack Name'] != 'MITM ARP Spoofing'].copy()

    attack_mapping = {
        'DDoS UDP Flood': 'UDP Flood',
        'DoS UDP Flood': 'UDP Flood',
        'DDoS ICMP Flood': 'ICMP Flood',
        'DoS ICMP Flood': 'ICMP Flood',
        'Recon OS Scan': 'Recon Scan',
        'Recon Port Scan': 'Recon Scan',
        'Recon Vulnerability Scan': 'Recon Scan'
    }

    dataframe_filtered['Attack Name'] = dataframe_filtered['Attack Name'].replace(attack_mapping)

    # Sample from each Attack Name to build smaller model for training
    small_df = dataframe_filtered.groupby('Attack Name').apply(sample_or_take_all)
    small_df = small_df.reset_index(level=0)

    # Separate target from features
    targets = small_df['Attack Name']
    # Removes any Strings from the Dataset (Including Attack Name, Flow ID,...)
    features = small_df.select_dtypes(include=[numpy.number])

    # Drops Label (derived feature that classifies a row as either benign or an attack)
    features = features.drop(columns = ["Label"])

    # Splitting off test data
    # returns features_train, features_test, targets_train, targets_test
    return train_test_split(features, targets, test_size=0.2, random_state=42)

    # return small_df