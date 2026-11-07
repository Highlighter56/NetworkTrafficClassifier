
# import libraries 
from pathlib import Path
import argparse
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# import data loading/cleaning and model evaluation
from data_cleaning import data_cleaning
import test_store_model_results as mr

# import training algorithms
from training_algorithms.log_reg import log_reg
from training_algorithms.dec_tre import dec_tre
# from svm import svm
# from PLACE_HOLDER import PLACE_HOLDER


def main():

	print("=========== Main ===========")

	args = parse_arguments()
	project_root = Path(__file__).resolve().parents[1]
	raw_data_path = get_local_path(project_root) if args.raw_data_path is None else Path(args.raw_data_path).expanduser().resolve()
	cleaned_folder_path = project_root / "data" / "cleaned_data"
	models_folder_path = project_root / "models"

	if not raw_data_path.is_dir():
		raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_path}")
	if not list(raw_data_path.glob("*.csv")):
		raise FileNotFoundError(f"No CSV files found in raw data directory: {raw_data_path}")

	# ======== Get Data ========
	print("Checking if Cleaned Data Exists")

	# ----- If Cleaned Data Exists -----
	x_train, x_test, y_train, y_test = "", "", "", ""
	cleaned_files = {
		"x_train": cleaned_folder_path / "x_train.csv",
		"x_test": cleaned_folder_path / "x_test.csv",
		"y_train": cleaned_folder_path / "y_train.csv",
		"y_test": cleaned_folder_path / "y_test.csv",
	}

	if not args.regenerate and all(file.exists() for file in cleaned_files.values()):
		print("Cleaned Data Exists, Loading...")
		x_train = pd.read_csv(cleaned_files["x_train"])
		x_test = pd.read_csv(cleaned_files["x_test"])
		y_train = pd.read_csv(cleaned_files["y_train"])
		y_test = pd.read_csv(cleaned_files["y_test"])

	# ----- If No Cleaned Data Exists -----
	else: 
		if args.regenerate:
			print("Re-Cleaning Data...")
		else:
			print("Cleaned Data does NOT Exists, Cleaning...")
		cleaned_folder_path.mkdir(parents=True, exist_ok=True)

		# Load, Clean, Split the Data
		x_train, x_test, y_train, y_test = data_cleaning(str(raw_data_path / "*.csv"))

		# Store DF's as files in cleaned_data
		x_train.to_csv(cleaned_files["x_train"], index=False)
		x_test.to_csv(cleaned_files["x_test"], index=False)
		y_train.to_csv(cleaned_files["y_train"], index=False)
		y_test.to_csv(cleaned_files["y_test"], index=False)

	print("Data Loaded")


	# ----- Standardization of X for models that need it -----
	print("Standardizing Inputs")
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)


	# =========== Building/Getting Models ===========

	# ----- Building Model Paths -----
	models_folder_path.mkdir(parents=True, exist_ok=True)

	# Model File Names
	lr = models_folder_path / "log_reg_model.pkl"
	dt = models_folder_path / "dec_tre.pkl"

	# Instantiating model variables
	log_reg_model = ""
	
	# --- Load / Train Models ---
	log_reg_model = load_train_model(x_train_scaled, y_train, lr, log_reg, args.regenerate)
	dec_tre_model = load_train_model(x_train, y_train, dt, dec_tre, args.regenerate)

	# =========== Model Validation ===========
	print("====== Model Results =======")

	# ----- Logistic Regression ---
	print("\n\n--- Logistic Regression Results ---")
	log_reg_results = mr.Model_Results(log_reg_model, x_test_scaled, y_test, "Logistic Regression")
	# display_results(log_reg_results)

	# ----- Decission Tree -----
	print("\n\n--- Decission Tree Results ---")
	dec_tree_results = mr.Model_Results(dec_tre_model, x_test, y_test, "Decission Tree")
	# display_results(dec_tree_results)
	
	# ----- SVM -----
	print("\n\n--- SVM Results ---")
	# svm_results = mr.Model_Results(svm_model, x_test_scaled, y_test, "SVM")
	# display_results(svm_results)

	# ----- PLACE_HOLDER -----
	print("\n\n--- PLACE_HOLDER Results ---")
	# PLACE_HOLDER_results = mr.Model_Results(PLACE_HOLDER_results, x_test, y_test, "PLACE_HOLDER")
	# display_results(PLACE_HOLDER_results)


	print("=========== Done ===========")


def parse_arguments():
	parser = argparse.ArgumentParser(description="Train and evaluate network traffic classifiers.")
	parser.add_argument(
		"raw_data_path",
		nargs="?",
		help="Directory containing raw CSV files. Defaults to data/raw_data in the project.",
	)
	parser.add_argument(
		"-i",
		"--invalidate",
		dest="regenerate",
		action="store_true",
		help="Regenerate cleaned data and retrain all enabled models.",
	)
	return parser.parse_args()


def get_local_path(project_root):
	return project_root / "data" / "raw_data"


def display_results(model):
	print("Accuracy:",model.get_accuracy())
	print("Model Report:",model.get_model_report())
	model.display_confusion_matrix_data()

def load_train_model(x_train, y_train, path, training_function, force_retrain=False):
	# If a Trained Model is Stored
	if Path(path).exists() and not force_retrain:
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