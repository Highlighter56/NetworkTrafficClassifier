# import numpy as np
# import sys
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Model Validation
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import model_test

# from train_logistic_regression import train_logistic_regression
# from data_cleaning_ian import data_cleaning
import data_cleaning_ian


def train_logistic_regression(x_train, y_train, iter):
	# Instantiate Model
	logistic_regression = LogisticRegression(
		random_state=42,
		max_iter = iter
	)

	# Train Model
	model = logistic_regression.fit(x_train, y_train)
	return model

def log_reg_nv(): 
	# === Loading, Cleaning, and Creating the Dataframe ===
	print("Creating Dataframe")
	print("Assigning Inputs and Output")
	print("Splitting Training and Testing Data")
	x_train, x_test, y_train, y_test = data_cleaning_ian.data_cleaning()

	# --- Standardization of X ---
	print("Standardizing Inputs")
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)


	# === Train Models on Specific Algorithms ===
	print("Training Model")
	model_lr = train_logistic_regression(x_train_scaled, y_train, 1000)
	# print("Model Trained")
	
	print("===== Model Results =====")
	results = model_test.Model_Test(model_lr, x_test_scaled, y_test)
	print("Accuracy:",results.get_accuracy())
	print("Model Report:",results.get_model_report())
	results.display_confusion_matrix_data()


if __name__ == "__main__":
	log_reg_nv()