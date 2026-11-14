
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

# Import evaluation report displays
from evaluation import evaluate, evaluate_class_report, plot_confusion_matrix, plot_confusion_matrix_norm

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

	# =========== Model Evaluation ===========
	# Evaluate models
	print("Evaluating Logistic Regression Classifier...", end=" ", flush=True)
	train_pred_lr = log_reg_model.predict(x_train_scaled)
	test_pred_lr = log_reg_model.predict(x_test_scaled)
	print("done")

	print("Evaluating Decision Tree Classifier...", end=" ", flush=True)
	train_pred_dt = dec_tre_model.predict(x_train)
	test_pred_dt = dec_tre_model.predict(x_test)
	print("done")

	print("Evaluating SVM Classifier...", end=" ", flush=True)
	train_pred_svm = svm_model.predict(X_train_kbest)
	test_pred_svm = svm_model.predict(X_test_kbest)
	print("done")

	print("Evaluating Random Forest Classifier...", end=" ", flush=True)
	train_pred_rf = rnd_for_model.predict(x_train)
	test_pred_rf = rnd_for_model.predict(x_test)
	print("done")

	print("====== Model Results =======")
	print("\n\n--- Logistic Regression Results ---")
	# Metric Reports
	report_lg = pd.DataFrame({
		"Training": evaluate(y_train, train_pred_lr),
		"Testing": evaluate(y_test, test_pred_lr)
	})
	print("\nOverall Results:")
	print(report_lg.round(4))

	# Classification Reports
	report_class_train_lg = pd.DataFrame(evaluate_class_report(y_train, train_pred_dt)).T
	report_class_test_lg = pd.DataFrame(evaluate_class_report(y_test, test_pred_dt)).T
	print("\n Classification Report Training:")
	print(report_class_train_lg.round(4))
	print("\n Classification Report Testing:")
	print(report_class_test_lg.round(4))

	# Confusion Matrices
	cm_train_lg = plot_confusion_matrix(y_train, train_pred_lr, log_reg_model.classes_, title="Confusion Matrix Logistic Regression (Train)")
	cm_train_lg_norm = plot_confusion_matrix_norm(y_train, train_pred_lr, log_reg_model.classes_, title="Confusion Matrix Logistic Regression Normalized (Train)")
	cm_test_lg = plot_confusion_matrix(y_test, test_pred_lr, log_reg_model.classes_, title="Confusion Matrix Logistic Regression (Test)")
	cm_test_lg_norm = plot_confusion_matrix_norm(y_test, test_pred_lr, log_reg_model.classes_, title="Confusion Matrix Logistic Regression Normalized (Test)")

	print("\n\n--- Decision Tree Results ---")
	# Metric Reports
	report_dt = pd.DataFrame({
		"Training": evaluate(y_train, train_pred_dt),
		"Testing": evaluate(y_test, test_pred_dt)
	})
	print("\nOverall Results:")
	print(report_dt.round(4))

	# Classification Reports
	report_class_train_dt = pd.DataFrame(evaluate_class_report(y_train, train_pred_dt)).T
	report_class_test_dt = pd.DataFrame(evaluate_class_report(y_test, test_pred_dt)).T
	print("\n Classification Report Training:")
	print(report_class_train_dt.round(4))
	print("\n Classification Report Testing:")
	print(report_class_test_dt.round(4))

	# Confusion Matrices
	cm_train_dt = plot_confusion_matrix(y_train, train_pred_dt, dec_tre_model.classes_, title="Confusion Matrix Decision Tree (Train)")
	cm_train_dt_norm = plot_confusion_matrix_norm(y_train, train_pred_dt, dec_tre_model.classes_, title="Confusion Matrix Decision Tree Normalized (Train)")
	cm_test_dt = plot_confusion_matrix(y_test, test_pred_dt, dec_tre_model.classes_, title="Confusion Matrix Decision Tree (Test)")
	cm_test_dt_norm = plot_confusion_matrix_norm(y_test, test_pred_dt, dec_tre_model.classes_, title="Confusion Matrix Decision Tree Normalized (Test)")

	print("\n\n--- SVM Results ---")
	# Metric Reports
	report_svm = pd.DataFrame({
		"Training": evaluate(y_train, train_pred_svm),
		"Testing": evaluate(y_test, test_pred_svm)
	})
	print("\nOverall Results:")
	print(report_svm.round(4))

	# Classification Reports
	report_class_train_svm = pd.DataFrame(evaluate_class_report(y_train, train_pred_svm)).T
	report_class_test_svm = pd.DataFrame(evaluate_class_report(y_test, test_pred_svm)).T
	print("\n Classification Report Training:")
	print(report_class_train_svm.round(4))
	print("\n Classification Report Testing:")
	print(report_class_test_svm.round(4))

	# Confusion Matrices
	cm_train_svm = plot_confusion_matrix(y_train, train_pred_svm, svm_model.classes_, title="Confusion Matrix SVM (Train)")
	cm_train_svm_norm = plot_confusion_matrix_norm(y_train, train_pred_svm, svm_model.classes_, title="Confusion Matrix SVM Normalized (Train)")
	cm_test_svm = plot_confusion_matrix(y_test, test_pred_svm, svm_model.classes_, title="Confusion Matrix SVM (Test)")
	cm_test_svm_norm = plot_confusion_matrix_norm(y_test, test_pred_svm, svm_model.classes_, title="Confusion Matrix SVM Normalized (Test)")

	print("\n\n--- Random Forest Results ---")
	# Metric Reports
	report_rf = pd.DataFrame({
		"Training": evaluate(y_train, train_pred_rf),
		"Testing": evaluate(y_test, test_pred_rf)
	})
	print("\n Overall Results:")
	print(report_rf.round(4))

	# Classification Reports
	report_class_train_rf = pd.DataFrame(evaluate_class_report(y_train, train_pred_rf)).T
	report_class_test_rf = pd.DataFrame(evaluate_class_report(y_test, test_pred_rf)).T
	print("\n Classification Report Training:")
	print(report_class_train_rf.round(4))
	print("\n Classification Report Testing:")
	print(report_class_test_rf.round(4))

	# Confusion Matrices
	cm_train_rf = plot_confusion_matrix(y_train, train_pred_rf, rnd_for_model.classes_, title="Confusion Matrix Random Forest (Train)")
	cm_train_rf_norm = plot_confusion_matrix_norm(y_train, train_pred_rf, rnd_for_model.classes_, title="Confusion Matrix Random Forest (Normalized)")
	cm_test_rf = plot_confusion_matrix(y_test, test_pred_rf, rnd_for_model.classes_, title="Confusion Matrix Random Forest (Test)")
	cm_test_rf_norm = plot_confusion_matrix_norm(y_test, test_pred_rf, rnd_for_model.classes_, title="Confusion Matrix Random Forest (Normalized)")
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