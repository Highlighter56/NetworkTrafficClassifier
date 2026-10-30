# import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

def log_reg(): 
	# === Loading, Cleaning, and Creating the Dataframe ===


	print("Creating Dataframe")
	df = data_cleaning_ian.data_cleaning()
	print("Dataframe Created")
	
	# sys.exit(1)

	# === Spliting Data ===
	print("Assigning Inputs and Output")

	# --- Input ---
	x = df.drop(
		columns = [
			"Attack Name", 
			"Label",
			"Flow ID",
			"Src IP",
			"Dst IP",
			"Timestamp"
		]
	)
	# print(x.head())
	'''Logistic Regression doesnt like Strinsg, so in addition
	to dropping output values, we need to drop strings.
	IP values may apply to these collected pieces of data,
	but in general they dont so should be dropped, same for time
	'''

	# --- Output ---
	y = df["Attack Name"]
	# print(y.head())

	print("Splitting Training and Testing Data")
	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=.2,
		random_state=42,
		stratify=y
	)

	# --- Standardization of X ---
	print("Standardizing Inputs")
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)


	# === Train Models on Specific Algorithms ===
	print("Training Model")
	model_lr = train_logistic_regression(x_train_scaled, y_train, 1000)
	print("Model Trained")

	# === Test Model ===
	print("Testing Model")
	model_predictions = model_lr.predict(x_test_scaled)


	# === Validate Model Predictions ===
	print("===Evaluating Model===")

	# Evaluate the model
	model_accuracy = accuracy_score(y_test, model_predictions)
	model_report = classification_report(y_test, model_predictions, zero_division=0)
	print(f"Accuracy score: {model_accuracy:.4f}")
	print("Classification Report:")
	print(model_report)

	# === Confusion Matrix ===

	cm = confusion_matrix(y_test, model_predictions)

	fig, ax = plt.subplots(figsize=(10, 10))
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_lr.classes_)
	disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)
	plt.title("Logistic Regression Confusion Matrix")
	plt.tight_layout()
	plt.show()



if __name__ == "__main__":
	log_reg()