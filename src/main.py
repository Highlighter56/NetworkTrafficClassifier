
# import libraries 
from pathlib import Path
import sys
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest, f_classif

# import data loading/cleaning and model evaluation
from data_cleaning import data_cleaning
import test_store_model_results as mr

# import training algorithms
from training_algorithms.log_reg import log_reg
from training_algorithms.dec_tre import dec_tre
from training_algorithms.svm import svm
from training_algorithms.random_forest import random_forest


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

	# --- Outputs are converted from Dataframes into Arrays ---
	# y_train = y_train.squeeze()
	# y_test = y_test.squeeze()


	# ----- Standardization of X for models that need it -----
	print("Standardizing Inputs")
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)

	# ----- Data preprocessing unique to SVM -----
    # Select kbest features, dropping number of features to 60
	selector = SelectKBest(score_func=f_classif, k=60)
	X_train_kbest = selector.fit_transform(x_train_scaled, y_train)
	X_test_kbest = selector.transform(x_test_scaled)

	# =========== Building/Getting Models ===========

	# ----- Building Path Variables -----
	# Base Path
	path += "/../../models/"

	# Model File Names
	lr = "log_reg_model.pkl"
	dt = "dec_tre.pkl"
	svm_path = "svm_model.pkl"
	rnd_for = "rnd_for_model.pkl"

	# Instantiating model variables
	log_reg_model = ""
	svm_model = ""
	dec_tre_model = ""
	rnd_for_model = ""
	
	# --- Load / Train Models ---
	log_reg_model = load_train_model(x_train_scaled, y_train, path + lr, log_reg)
	dec_tre_model = load_train_model(x_train, y_train, path + dt, dec_tre)
	svm_model = load_train_model(X_train_kbest, y_train, path + svm_path, svm)
	rnd_for_model = load_train_model(x_train, y_train, path + rnd_for, random_forest)

	# =========== Model Validation ===========
	print("====== Model Results =======")

	# ----- Logistic Regression ---
	print("\n\n--- Logistic Regression Results ---")
	log_reg_results = mr.Model_Results(log_reg_model, x_test_scaled, y_test, "Logistic Regression")
	display_results(log_reg_results)

	# ----- Decission Tree -----
	print("\n\n--- Decission Tree Results ---")
	dec_tree_results = mr.Model_Results(dec_tre_model, x_test, y_test, "Decision Tree")
	display_results(dec_tree_results)
	
	# ----- SVM -----
	print("\n\n--- SVM Results ---")
	svm_results = mr.Model_Results(svm_model, X_test_kbest, y_test, "SVM")
	display_results(svm_results)

	# ----- Random Forest -----
	print("\n\n--- Random Forest Results ---")
	rnd_for_results = mr.Model_Results(rnd_for_model, x_test, y_test, "Random Forest")
	display_results(rnd_for_results)



	print("=========== Done ===========")

def display_results(model):
	print("Accuracy:",model.get_accuracy())
	print("Model Report:\n"+model.get_model_report())
	model.display_confusion_matrix_data()

def load_train_model(x_train, y_train, path, training_function):
	# If a Trained Model is Stored
	if Path(path).exists():
		print("Model Exists")
		print("Loading Model")
		with open(path, "rb") as file:
			return pickle.load(file)
	# If a Model needs to be Trained and Stored
	print("Model does not Exists")
	print("Training Model")
	return training_function(x_train, y_train, path)
	


if __name__ == "__main__":
	main()