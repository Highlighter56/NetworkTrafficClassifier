from sklearn.linear_model import LogisticRegression


def train_logistic_regression(x_train, y_train, iter):
	# Instantiate Model
	logistic_regression = LogisticRegression(
		random_state=42,
		max_iter = iter
	)

	# Train Model
	model = logistic_regression.fit(x_train, y_train)
	return model



if __name__ == "__main__":
	train_logistic_regression()