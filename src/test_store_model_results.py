
# Needed to import .predict
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay



class Model_Results:
	# Constructor
	def __init__(self, trained_model, x_test, y_test, title):
		# Saving the Input Parameters
		self.model = trained_model
		self.x_test_scaled = x_test
		self.y_test = y_test
		self.title = title

		# Intermidiate Calculations
		self.model_predictions = self._test_model(trained_model, x_test)

		# Final Desired Results
		self.accuracy = self._calculate_accuracy(y_test, self.model_predictions)
		self.model_report = self._calculate_classification_report(y_test, self.model_predictions)
		self.confusion_matrix_data = self._generate_confusion_matrix(y_test, self.model_predictions)

	# Getters
	def get_accuracy(self, truncated=False):
		if truncated == True:
			return self.accuracy
		# Easy and weird way to truncate a float 
		return float(f"{self.accuracy:.4f}")

	def get_model_report(self):
		return self.model_report

	def get_confusion_matrix_data(self):
		return self.confusion_matrix_data

	def display_confusion_matrix_data(self):
		fig, ax = plt.subplots(figsize=(10, 10))

		disp = ConfusionMatrixDisplay(
			confusion_matrix=self.confusion_matrix_data,
			display_labels=self.model.classes_
		)

		disp.plot(
			ax=ax,
			xticks_rotation='vertical',
			cmap=plt.cm.Blues
		)

		ax.set_title(self.title + " Confusion Matrix")
		fig.tight_layout()

		plt.show()

	# Calculations
	def _test_model(self, model, x_test):
		return model.predict(x_test)
	
	def _calculate_accuracy(self, y_test, model_predictions):
		return accuracy_score(y_test, model_predictions)
	
	def _calculate_classification_report(self, y_test, model_predictions):
		return classification_report(y_test, model_predictions, zero_division=0)
	
	def _generate_confusion_matrix(self, y_test, model_predictions):
		return confusion_matrix(y_test, model_predictions)