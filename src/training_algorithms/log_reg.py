
# Logistic Regression

from sklearn.linear_model import LogisticRegression
import pickle



def log_reg(x_train_scaled, y_train, path): 


	# --- Instantiate Model ---
	print("Training Logistic Regression Model")

	logistic_regression = LogisticRegression(
			max_iter=5000,
			C = 100,
			class_weight=None,
			penalty="l2",
			solver="lbfgs"
		)
	
	# --- Train Model ---
	model_lr = logistic_regression.fit(x_train_scaled, y_train)

	# --- Store Model ---
	print("Storing Model")
	with open(path, "wb") as file:
		pickle.dump(model_lr, file)

	return model_lr



if __name__ == "__main__":
	log_reg()