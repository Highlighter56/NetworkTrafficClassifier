import pickle

from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train, y_train, path):

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
    random_forest_model = random_forest_clf.fit(X_train, y_train)

    # Store model in pickle file
    with open(path, 'wb') as file:
        pickle.dump(random_forest_model, file)

    return random_forest_model

if __name__ == "__main__":
    random_forest()
