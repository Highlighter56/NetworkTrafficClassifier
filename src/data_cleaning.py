# Python script for cleaning data files
import glob
import pandas as pandas

csv_folder = "../data/loose_csv/*.csv"
file_paths = glob.glob(csv_folder)

print(f"Found {len(file_paths)} files")

dataframe_list = [pandas.read_csv(file) for file in file_paths]
dataframe_concat = pandas.concat(dataframe_list, ignore_index=True)

dataframe_concat = dataframe_concat.reset_index(drop=True)
print(dataframe_concat.head())

small_df = dataframe_concat.groupby('Attack Name').sample(n=100, replace=False, random_state=42)
small_df = small_df.reset_index(drop=True)

print(small_df.shape)
print(small_df.head())