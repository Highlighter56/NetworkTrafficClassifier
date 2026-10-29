# Separate data cleaning script to make merging easier
import glob
import pandas as pandas

# Load all csv files in dataset folder
csv_folder = "../data/loose_csv/*.csv"
file_paths = glob.glob(csv_folder)

# Build a dataframe for each csv file and store in list
dataframe_list = [pandas.read_csv(file) for file in file_paths]

# Concatenate each dataframe in list and reset index
dataframe_combined = pandas.concat(dataframe_list, ignore_index=True)
dataframe_combined = dataframe_combined.reset_index(drop=True)