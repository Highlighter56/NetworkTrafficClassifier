# Helper functions for hold-out validation reports
import csv
import argparse
import os
import pickle

import pandas as pd

from datetime import datetime, timezone
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

# Report Formatting and Writing to CSV
def write_eval_to_csv(results: dict) -> None:
    with open("results.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        results['created_at'] = datetime.now(timezone.utc)
        writer.writerow(results)

# Start here
# Get model
print("Loading model...", end=' ', flush=True)

parser = argparse.ArgumentParser(description='Evaluate model using holdout validation.')

parser.add_argument('model', type=str, help='Filename of model pickle file')
args = parser.parse_args()
model_name = args.model

current_folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_folder, 'training_models')

os.makedirs(model_folder, exist_ok=True)
path = os.path.join(model_folder, 'random_forest_holdout.pkl')

with open(path, 'rb') as file:
    model = pickle.load(file)

print('done')

# Get cleaned data
print('Loading data...', end=' ', flush=True)

features = pd.read_csv('../../data/holdout_data/X_val_holdout.csv')
targets = pd.read_csv('../../data/holdout_data/y_val_holdout.csv').squeeze('columns')

print('done.')

print('Evaluating model...', end=' ', flush=True)
prediction = model.predict(features)

results_metrics = evaluate(targets, prediction)
results_classification_report = evaluate_class_report(targets, prediction)
cm = plot_confusion_matrix(targets, prediction, model.classes_, title="Confusion Matrix")
cm_norm = plot_confusion_matrix_norm(targets, prediction, model.classes_, title="Normalized Confusion Matrix")

print('done.')
print(results_metrics)
print(results_classification_report)
print('Evaluation complete')
