"""visualization.py

Visualization helpers for model performance.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    """Plot a confusion-matrix heatmap given true and predicted labels."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# TRAINING CURVE
# ---------------------------------------------------------
def plot_training(history):
    """Plot training and validation accuracy curves from a Keras History."""
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="Train")

    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation")

    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
