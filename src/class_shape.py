# Python script for cleaning data files
import glob
import sys
import pandas as pandas
import numpy as numpy
from sklearn.model_selection import train_test_split


# 
def load_csv_data():
    path = ""

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


def data_cleaning():
    # Store the loose csv files into list

    file_paths = load_csv_data()

    # Build a dataframe for each csv file and store in list
    dataframe_list = [pandas.read_csv(file) for file in file_paths]

	
    i = 0
    for type in dataframe_list:
        print(i , type.shape)
        i+=1
