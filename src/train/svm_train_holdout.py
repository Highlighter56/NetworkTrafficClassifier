import os
import pickle
import time
import pandas as pd

from sklearn.svm import SVC

# Get cleaned data
print('Loading data...', end=' ', flush=True)

features = pd.read_csv('../../data/holdout_data/X_train_holdout.csv')
targets = pd.read_csv('../../data/holdout_data/y_train_holdout.csv').squeeze('columns')

print('done.')

# Build SVM Classifier
svc_clf = SVC(kernel='rbf', C=100, gamma='auto', class_weight=None)

# Train model
print('Training random forest classifier...', end=' ', flush=True)
start_time = time.time()

svm_model = svc_clf.fit(features, targets)

elapsed_time = time.time() - start_time
print(f'done. Trained in {elapsed_time:.2f} seconds.')

# Store model in pickle file
print('Saving model...', end=' ', flush=True)

current_folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_folder, 'training_models')

os.makedirs(model_folder, exist_ok=True)
path = os.path.join(model_folder, 'svm_train_holdout.pkl')

with open(path, 'wb') as file:
    pickle.dump(svm_model, file)

print('done')
print(f'Model saved to {path}')
print("Training complete.")
