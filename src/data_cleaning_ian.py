# Python script for cleaning data files
import glob
import sys
import pandas as pandas

def load_csv_data():
	path = ""

	if len(sys.argv) == 2:
		path = sys.argv[1] + "*.csv"

	if path == "":
		path = "../data/loose_csv/*.csv"

	return glob.glob(path)

def sample_or_take_all(group):
    sample_size = min(len(group), 1000)
    return group.sample(n=sample_size, random_state=42)

def data_cleaning():
    # Store the loose csv files into list
        
    file_paths = load_csv_data()

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

    return small_df

