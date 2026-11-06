
from pathlib import Path
import sys
import glob
import pandas as pd


# from log_reg import log_reg
from data_cleaning_ian import data_cleaning
from log_reg_nv import log_reg_nv

def main():

	print("=========== Main ===========")

	# ======== Get Data ========
	path = ""

	# If Entered too many args
	if len(sys.argv) > 2:
		print("---Error---")
		print("Too many Argumenets Entered")
		print("Expected One at Most")
		print("Please refer to README.md")
		sys.exit(1)
	
	# If Entered a custom path
	elif len(sys.argv) == 2:
		path = sys.argv[1] + "../cleaned_data"
	# If no args entered
	else: 
		file_pathsA = glob.glob("../data/raw_data/*.csv")
		if len(file_pathsA) == 15:
			path = "../data/cleaned_data"
		else:
			path = "data/cleaned_data"

	cleaned_folder_path = Path(path)

	x_train, x_test, y_train, y_test = "", "", "", ""

	# =========== If Cleaned Data Exists ==========
	if cleaned_folder_path.is_dir():
		print("Cleaned Data Exists")
		x_train = pd.read_csv(path+"/x_train.csv")
		x_test = pd.read_csv(path+"/x_test.csv")
		y_train = pd.read_csv(path+"/y_train.csv")
		y_test = pd.read_csv(path+"/y_test.csv")

	# ============ If No Cleaned Data Exists ==========
	else: 
		print("Cleaned Data does NOT Exist")
		# Create Directory cleaned_data
		cleaned_folder_path.mkdir()

		# Load, Clean, Split the Data
		x_train, x_test, y_train, y_test = data_cleaning()

		# Store DF's as files in cleaned_data
		x_train.to_csv(path+"/x_train.csv", index=False)
		x_test.to_csv(path+"/x_test.csv", index=False)
		y_train.to_csv(path+"/y_train.csv", index=False)
		y_test.to_csv(path+"/y_test.csv", index=False)


	print("Data Loaded")



	# model = log_reg_nv()




	print("=========== Done ===========")




if __name__ == "__main__":
	main()