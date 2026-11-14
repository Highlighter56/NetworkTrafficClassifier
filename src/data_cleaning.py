# Python script for cleaning data files
import glob
import sys
import pandas as pandas

from sklearn.model_selection import train_test_split

# Loads all csv files from a folder
def load_csv_data():
    # If entered a custom path
    if len(sys.argv) == 2:
        return glob.glob(sys.argv[1] + "*.csv")

    # If Entered no Path
    # If running from scr
    file_pathsA = glob.glob("../data/raw_data/*.csv")
    if len(file_pathsA) == 15:
          return file_pathsA
    # If running from root
    return glob.glob("data/raw_data/*.csv")


def sample_or_take_all(group):
    sample_size = min(len(group), 6000)
    return group.sample(n=sample_size, random_state=42)

def data_cleaning():
    # Store the loose csv files into list

    file_paths = load_csv_data()
    print(f"Found {len(file_paths)} files.")

    # Testing you can run from src, root, or a custom destination provided you specify where the data is
    # print("File Path Size:",len(file_paths))
    # sys.exit(1)

    # Build a dataframe for each csv file and store in list
    dataframe_list = [pandas.read_csv(file) for file in file_paths]

    # Concatenate each dataframe in list and reset index
    dataframe_combined = pandas.concat(dataframe_list, ignore_index=True)
    dataframe_combined = dataframe_combined.reset_index(drop=True)

    # print("Shape No Drop: ",dataframe_combined.shape)
    # sys.exit(1)
    
    # print("Shape With Drop: ",dataframe_combined.shape)
    # sys.exit(1)

    # Drop problematic attack types (Recon Ping Sweep, MITM ARP Spoofing)
    dataframe_filtered = dataframe_combined[~dataframe_combined['Attack Name'].isin(['Recon Ping Sweep', 'MITM ARP Spoofing'])].copy()

    # Replace space characters with "_"
    dataframe_filtered.columns = dataframe_filtered.columns.str.replace(" ", "_")

    # Drop problematic columns
    dataframe_filtered = dataframe_filtered.drop(columns=["Flow_ID", "Src_IP", "Src_Port", "Dst_IP", "Dst_Port", "Protocol", "Timestamp", "Label"])

    # Drop rows will null values
    dataframe_filtered.dropna()

    # Sample from each Attack Name to build smaller model for training
    small_df = dataframe_filtered.groupby("Attack_Name").apply(sample_or_take_all)
    small_df = small_df.reset_index(level=0)

    # Shuffle small_df
    small_df_rnd = small_df.sample(frac=1, random_state=42)
    small_df_rnd = small_df_rnd.reset_index(drop=True)

    # Separate target from features
    features = small_df_rnd.drop(columns=["Attack_Name"])
    targets = small_df_rnd["Attack_Name"]

    # Splitting off test data (80/20 split)
    # returns features_train, features_test, targets_train, targets_test
    return train_test_split(features, targets, test_size=0.2, random_state=42)
    
if __name__ == "__main__":
    data_cleaning()