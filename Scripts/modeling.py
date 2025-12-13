"""modeling.py

Model training utilities for classical ML models
(Logistic Regression, SVM, Random Forest, XGBoost)
and a simple feed-forward neural network.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models


# ---------------------------------------------------------
# CLASSICAL MACHINE-LEARNING MODELS
# ---------------------------------------------------------
def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier


def train_svm(X_train, y_train):
    """Train a linear Support Vector Machine classifier."""
    classifier = SVC(kernel="linear", C=1.0, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    return classifier


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    return classifier


# ---------------------------------------------------------
# DEEP LEARNING MODEL (FEED-FORWARD NETWORK)
# ---------------------------------------------------------
def build_ffnn(input_dim: int):
    """Build a simple feed-forward neural network (binary classifier)."""
    network = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    network.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return network


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test, name: str = "Model"):
    """Print accuracy and a classification report for a trained model."""
    y_pred = model.predict(X_test)
    print(f"\n===== {name} Evaluation =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# ---------------------------------------------------------
# PERSISTENCE HELPERS
# ---------------------------------------------------------
def save_model(model, path: str):
    """Persist a trained model using joblib."""
    joblib.dump(model, path)
    print(f"[INFO] Saved model to: {path}")
