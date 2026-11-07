
# Decission Tree

from sklearn.tree import DecisionTreeClassifier
import pickle



def dec_tre(x_train, y_train, path): 


	# --- Instantiate Model ---
	print("Training Decission Tree Model")

	# - Parameters to Play With -
	# max_depth = 20
	# min_samples_split = 10
	# criterion = "entropy"
	decision_tree = DecisionTreeClassifier(
			random_state=42,
			max_depth = 20,
			min_samples_split = 10,
			criterion = "entropy"
		)
	
	# --- Train Model ---
	model_dt = decision_tree.fit(x_train, y_train)

	# --- Store Model ---
	print("Storing Model")
	with open(path, "wb") as file:
		pickle.dump(model_dt, file)

	return model_dt



if __name__ == "__main__":
	dec_tre()