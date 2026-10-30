# Python script for cleaning data files
import glob
import pandas as pandas



def create_df(file_paths): 

	# === Load Data ===
	dataframe_list = [pandas.read_csv(file) for file in file_paths]

	# === Form Entire Data Frame
	df = pandas.concat(dataframe_list, ignore_index=True)
	# print("Dataframe.shape:", df.shape)

	return df


if __name__ == "__main__":
	create_df()