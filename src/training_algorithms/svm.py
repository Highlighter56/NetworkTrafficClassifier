import pickle

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

def svm(x_train, y_train, path):
    # Data preprocessing unique to SVM
    # Select kbest features, dropping number of features to 60
    selector = SelectKBest(score_func=f_classif, k=60)
    X_train_kbest = selector.fit_transform(x_train, y_train)

    # Build SVM Classifier
    svc = SVC(
        kernel='rbf',
        C=100,
        gamma='auto',
        class_weight='None',
    )

    # Train Model
    svm_model = svc.fit(X_train_kbest, y_train)

    # Store model in pickle file
    with open(path, "wb") as file:
        pickle.dump(svm_model, file)

    return svm_model

if __name__ == "__main__":
    svm()