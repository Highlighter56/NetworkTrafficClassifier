import pandas as pandas
import numpy as numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

from data_cleaning_ian import data_cleaning

# Load cleaned dataframe
small_df = data_cleaning()

# Separate target from features
targets = small_df['Attack Name']
features = small_df.select_dtypes(include=[numpy.number])

# Divide training and testing
features_train, features_test, target_train, target_test = train_test_split(features, targets, test_size=0.2, random_state=42, stratify=targets)

var_selector = VarianceThreshold(threshold=0.0)

features_train_clean = var_selector.fit_transform(features_train)
features_test_clean = var_selector.transform(features_test)

# Scale features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_clean)
features_test_scaled = scaler.transform(features_test_clean)

# Apply selector SelectKBest k=60
selector = SelectKBest(score_func=f_classif, k=60)
features_train_kbest = selector.fit_transform(features_train_scaled, target_train)
features_test_kbest = selector.transform(features_test_scaled)

# Determine parameters and train model
param_grid = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced'],
}

svc = SVC()
grid_search = GridSearchCV(svc, param_grid)
grid_search.fit(features_train_kbest, target_train)

print("best parameters:")
print(grid_search.best_params_)
print("Best Cross-Validation Score:")
print(grid_search.best_score_)

# Evaluate Model
target_prediction = grid_search.predict(features_test_kbest)
print(f"Accuracy score: {accuracy_score(target_test, target_prediction):.4f}")
print("Classification Report:")
print(classification_report(target_test, target_prediction, zero_division=0))
