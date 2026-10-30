# Python script for cleaning data files
import glob
import pandas as pandas



'''
Loads and cleans the data
Optional 1 paremeter: selective_size
if entered, rows of data pulled from each individual attack type will be 
limited to that cap
'''
def clean_data(selective_size = None): 

	print("Selective Size:",selective_size)

	# =========================================
	# If running from inside source
	# csv_folder = "../data/loose_csv/*.csv"
	# If running from root
	csv_folder = "data/loose_csv/*.csv"

	file_paths = glob.glob(csv_folder)

	print(f"Found {len(file_paths)} files")

	dataframe_list = [pandas.read_csv(file) for file in file_paths]
	# print("=============")
	# print(dataframe_list)
	# print("=============")
	# ==========================================

	# dataframe_concat is all the csv files grouped into one large dataframe
	dataframe_concat = pandas.concat(dataframe_list, ignore_index=True)
	dataframe_concat = dataframe_concat.sample(frac=1, random_state=42)
	dataframe_concat = dataframe_concat.reset_index(drop=True)
	print("Dataframe.shape:", dataframe_concat.shape)

	# If no selective size was entered, return entire dataframe
	if(selective_size is None):
		return dataframe_concat
	# Else restrict the size and continue

	# Breaking up the Larger Dataframe into a smaller selective version
	selective_df = dataframe_concat.groupby('Attack Name').sample(n=selective_size, replace=True, random_state=42)
	selective_df = selective_df.reset_index(drop=True)

	print("Selected Dataframe:", selective_df.shape)
	print(selective_df.head())

	# Return the size restricted / selective dataframe
	return selective_df



if __name__ == "__main__":
	clean_data()