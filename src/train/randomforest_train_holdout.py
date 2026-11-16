import os
import pickle
import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Get cleaned data
print('Loading data...', end=' ', flush=True)

features = pd.read_csv('../../data/holdout_data/X_train_holdout.csv')
targets = pd.read_csv('../../data/holdout_data/y_train_holdout.csv').squeeze('columns')

print('done.')

# Build Random Forest Classifier
random_forest_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced'
)

# Train model
print('Training random forest classifier...', end=' ', flush=True)
start_time = time.time()

random_forest_model = random_forest_clf.fit(features, targets)

elapsed_time = time.time() - start_time
print(f'done. Trained in {elapsed_time:.2f} seconds.')

# Store model in pickle file
print('Saving model...', end=' ', flush=True)

current_folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_folder, 'training_models')

os.makedirs(model_folder, exist_ok=True)
path = os.path.join(model_folder, 'random_forest_holdout.pkl')

with open(path, 'wb') as file:
    pickle.dump(random_forest_model, file)

print('done')
print(f'Model saved to {path}')
print("Training complete.")

