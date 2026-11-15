
# Decission Tree

from sklearn.tree import DecisionTreeClassifier
import pickle



def dec_tre(x_train, y_train, path): 


	# --- Instantiate Model ---
	print("Training Decission Tree Model")



	decision_tree = DecisionTreeClassifier(
			criterion = "gini",
			max_depth = 40,
			min_samples_split = 2,
			min_samples_leaf = 1,
			max_features = None,
			class_weight = "balanced"
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