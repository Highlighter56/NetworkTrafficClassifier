
from pathlib import Path
import sys
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


# from log_reg import log_reg
from data_cleaning import data_cleaning
from log_reg import log_reg
import test_store_model_results as mr

def main():

	print("=========== Main ===========")

	# ======== Get Data ========
	print("Checking if Cleaned Data Exists")

	# ----- Building Path -----
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
		# If running from inside src
		file_pathsA = glob.glob("../data/raw_data/*.csv")
		if len(file_pathsA) == 15:
			path = "../data/cleaned_data"
		# If running from root
		else:
			path = "data/cleaned_data"

	cleaned_folder_path = Path(path)


	# ----- If Cleaned Data Exists -----
	x_train, x_test, y_train, y_test = "", "", "", ""

	if cleaned_folder_path.is_dir():
		print("Cleaned Data Exists, Loading...")
		x_train = pd.read_csv(path+"/x_train.csv")
		x_test = pd.read_csv(path+"/x_test.csv")
		y_train = pd.read_csv(path+"/y_train.csv")
		y_test = pd.read_csv(path+"/y_test.csv")

	# ----- If No Cleaned Data Exists -----
	else: 
		print("Cleaned Data does NOT Exist, Gathering...")
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


	# ----- Standardization of X for models that need it -----
	print("Standardizing Inputs")
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)


	# =========== Building/Getting Models ===========

	# ----- Building Path Variables -----
	# Base Path
	path += "../../../models/"

	# Model File Names
	lr = "log_reg_model.pkl"
	svm = "svm_model.plk"
	dt = "dec_tre.plk"
	PLACE_HOLDER = "PLACE_HOLDER.plk"

	# Instantiating model variables
	log_reg_model = ""
	svm_model = ""
	dt_model = ""
	PLACE_HOLDER_model = ""
	
	# --- Load / Train Models ---
	log_reg_model = load_train_model(x_test_scaled, y_test, path + lr)
	svm_model = load_train_model(x_test_scaled, y_test, path + svm)

	# =========== Model Validation ===========
	print("====== Model Results =======")

	# ----- Logistic Regression ---
	print("\n\n--- Logistic Regression Results ---")
	log_reg_results = mr.Model_Results(log_reg_model, x_test_scaled, y_test)
	print("Accuracy:",log_reg_results.get_accuracy())
	print("Model Report:",log_reg_results.get_model_report())
	log_reg_results.display_confusion_matrix_data()

	# ----- SVM -----
	print("\n\n--- SVM Results ---")
	svm_results = mr.Model_Results(svm_model, x_test_scaled, y_test)
	print("Accuracy:",svm_results.get_accuracy())
	print("Model Report:",svm_results.get_model_report())
	svm_results.display_confusion_matrix_data()

	# ----- Decission Tree -----
	print("\n\n--- Decission Tree Results ---")

	# ----- PLACE_HOLDER -----
	print("\n\n--- PLACE_HOLDER Results ---")



	print("=========== Done ===========")

def load_train_model(x_train, y_train, path):
	# If a Trained Model is Stored
	if Path(path).exists():
		print("Model Exists")
		print("Loading Model")
		with open(path, "rb") as file:
			return pickle.load(file)
	# If a Model needs to be Trained and Stored
	print("Model does not Exists")
	print("Training Model")
	return log_reg(x_train, y_train, path)
	


if __name__ == "__main__":
	main()