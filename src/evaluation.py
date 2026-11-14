# Helper functions for hold-out validation reports

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Overall metrics
def evaluate(targets_data, prediction):
    return {
        "Accuracy": accuracy_score(targets_data, prediction),
        "Precision": precision_score(targets_data, prediction, average="weighted", zero_division=0),
        "Recall": recall_score(targets_data, prediction, average="weighted", zero_division=0),
        "F1": f1_score(targets_data, prediction, average="weighted", zero_division=0),
        "F1_macro": f1_score(targets_data, prediction, average="macro", zero_division=0),
    }

# Classification report
def evaluate_class_report(targets_data, prediction):
    return classification_report(targets_data, prediction, output_dict=True, zero_division=0)

# Confusion Matrix with raw counts
def plot_confusion_matrix(targets_data, prediction, classes, title="Confusion Matrix"):
    cm = confusion_matrix(targets_data, prediction, labels=classes)

    fig, ax = plt.subplots(figsize=(10,10))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    display.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cm

# Confusion Matrix with normalized counts
def plot_confusion_matrix_norm(targets_data, prediction, classes, title="Confusion Matrix"):
    cm_normalized = confusion_matrix(targets_data, prediction, labels=classes, normalize='true')

    fig, ax = plt.subplots(figsize=(10,10))
    display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)

    display.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', values_format=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cm_normalized