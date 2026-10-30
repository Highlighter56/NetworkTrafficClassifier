import glob
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from create_df import create_df
from train_logistic_regression import train_logistic_regression

def main():
	print("Starting Main")

	# ==== Load Data ==== 
	print("Getting Paths")
	# --- Finding CSV Files ---
	file_paths = get_csv_file_path()

	# --- Form Date Frame ---
	print("Creating Dataframe")
	df = create_df(file_paths)
	
	# print("Dataframe Created:",df.shape)


	# === Spliting Data ===
	print("Splitting Data")

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
	to dropping output values, we need to drop strings
	IP values may apply to these collected pieces of data,
	but in general they might not so should be removed. Same
	with time.'''
	# print(x.select_dtypes(exclude="number").columns.tolist())

	# --- Output ---
	y = df["Attack Name"]
	# print(y.head())

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=.2,
		random_state=42,
		stratify=y
	)

	# --- Standardization of X ---
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)


	# === Train Models on Specific Algorithms ===
	print("Training Model")
	model_lr = train_logistic_regression(x_train_scaled, y_train, 600)
	print("Model Trained")
	
	print("Testing Model")
	
	# === Test Model ===
	modelPredictions = model_lr.predict(x_test_scaled)

	#  ==== Validate Results ==== 
	print("Validate Results")

	# --- Count Correct Predictions ---
	correctPredictionCount = np.sum(y_test == modelPredictions)

	# --- Calculate Percent Correct ---
	accuracy = correctPredictionCount / modelPredictions.size
	
	print("Accuracy:",accuracy)



def get_csv_file_path():
	# If running from root
	csv_folder_root = "data/loose_csv/*.csv"
	path_a = glob.glob(csv_folder_root)
	
	# If running from inside src
	csv_folder_src = "../data/loose_csv/*.csv"
	path_b = glob.glob(csv_folder_src)

	# Assign file_paths
	# file_paths = None
	if(len(path_a) > len(path_b)):
		return path_a
	return path_b



class SplitData:
	def __init__(self, x_train, x_test, y_train, y_test):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test



if __name__ == "__main__":
	main()