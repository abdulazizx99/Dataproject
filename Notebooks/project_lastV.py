import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sentence_transformers import SentenceTransformer
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model as KerasModel


def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.show()


def save_all_models(models_dict: dict, save_dir: str = "models") -> None:
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model_obj in models_dict.items():

        # Case 1 — Keras deep learning model
        if isinstance(model_obj, KerasModel):
            file_path = os.path.join(save_dir, f"{model_name}.h5")
            model_obj.save(file_path)
            print(f"[Saved] Keras model -> {file_path}")

        # Case 2 — XGBoost model: best saved as json
        elif hasattr(model_obj, "save_model") and "xgb" in model_name.lower():
            file_path = os.path.join(save_dir, f"{model_name}.json")
            model_obj.save_model(file_path)
            print(f"[Saved] XGBoost model -> {file_path}")

        # Case 3 — Pickle-compatible models (Sklearn, etc.)
        else:
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model_obj, file_path)
            print(f"[Saved] Pickle model -> {file_path}")

    print("\nAll models saved successfully!")


def main(args: argparse.Namespace) -> None:
    # ----------------------------
    # Load df (CSV or Parquet)
    # ----------------------------
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        raise ValueError("Supported input formats: .csv or .parquet")

    # Basic checks
    required_cols = ["abstract_text_clean", "label"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Ensure types
    df["abstract_text_clean"] = df["abstract_text_clean"].astype(str)
    df["label"] = df["label"].astype(int)

    print("Loaded df:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())

    # ----------------------------
    # Split (Train/Val/Test) with stratify
    # ----------------------------
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        shuffle=True,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
        stratify=temp_df["label"]
    )

    print("\nSizes:")
    print("TOTAL:", len(df))
    print("TRAIN:", len(train_df))
    print("VAL:", len(val_df))
    print("TEST:", len(test_df))

    # ----------------------------
    # TF-IDF
    # ----------------------------
    tfidf_vectorer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        analyzer="word"
    )
    tfidf_vectorer.fit(train_df["abstract_text_clean"])

    X_train_tfidf = tfidf_vectorer.transform(train_df["abstract_text_clean"])
    X_val_tfidf   = tfidf_vectorer.transform(val_df["abstract_text_clean"])
    X_test_tfidf  = tfidf_vectorer.transform(test_df["abstract_text_clean"])

    print("\nTF-IDF shapes:")
    print("Train:", X_train_tfidf.shape)
    print("Validation:", X_val_tfidf.shape)
    print("Test:", X_test_tfidf.shape)

    # ----------------------------
    # Numeric features (your engineered features)
    # ----------------------------
    EXCLUDED_COLS = [
        "label", "abstract_text", "abstract_text_clean",
        "tokens", "words", "sentences", "paragraphs", "abstract_text_pos_tags"
    ]

    numeric_cols = [
        col for col in train_df.select_dtypes(include=np.number).columns
        if col not in EXCLUDED_COLS
    ]
    print("\nNumeric feature columns:", len(numeric_cols))

    X_train_num = csr_matrix(train_df[numeric_cols].to_numpy()) if numeric_cols else csr_matrix((len(train_df), 0))
    X_val_num   = csr_matrix(val_df[numeric_cols].to_numpy())   if numeric_cols else csr_matrix((len(val_df), 0))
    X_test_num  = csr_matrix(test_df[numeric_cols].to_numpy())  if numeric_cols else csr_matrix((len(test_df), 0))

    y_train = train_df["label"].to_numpy()
    y_val   = val_df["label"].to_numpy()
    y_test  = test_df["label"].to_numpy()

    X_train = hstack([X_train_tfidf, X_train_num]).tocsr()
    X_val   = hstack([X_val_tfidf,   X_val_num]).tocsr()
    X_test  = hstack([X_test_tfidf,  X_test_num]).tocsr()

    print("\nX and y ready:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    # ----------------------------
    # Logistic Regression
    # ----------------------------
    lr_model = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear")
    lr_model.fit(X_train, y_train)

    y_val_pred = lr_model.predict(X_val)
    print("\n[LR] Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    y_test_pred = lr_model.predict(X_test)
    print("\n[LR] Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm, "Confusion Matrix - Logistic Regression")

    # ----------------------------
    # SVM / RandomForest / XGBoost
    # ----------------------------
    models_dict = {"lr_model": lr_model}

    svm_model = LinearSVC(C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    models_dict["svm"] = svm_model

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models_dict["random_forest"] = rf_model

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models_dict["xgb_model"] = xgb_model

    # Validation check
    for name, model in models_dict.items():
        yv = model.predict(X_val)
        print(f"\n[{name}] Validation Accuracy:", accuracy_score(y_val, yv))
        print(classification_report(y_val, yv))

    # Test evaluation
    for name, model in models_dict.items():
        yt = model.predict(X_test)
        print(f"\n===== {name} Test Evaluation =====")
        print("Accuracy:", accuracy_score(y_test, yt))
        print(classification_report(y_test, yt))
        cm = confusion_matrix(y_test, yt)
        plot_confusion_matrix(cm, f"Confusion Matrix - {name}")

    # ----------------------------
    # Deep Learning (optional): Embeddings + FFNN
    # ----------------------------
    if args.run_dl:
        print("\n[DL] Loading SentenceTransformer and encoding...")
        st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        X_train_emb = st_model.encode(train_df["abstract_text_clean"].tolist(), convert_to_numpy=True)
        X_val_emb   = st_model.encode(val_df["abstract_text_clean"].tolist(), convert_to_numpy=True)
        X_test_emb  = st_model.encode(test_df["abstract_text_clean"].tolist(), convert_to_numpy=True)

        ffnn_model = models.Sequential([
            layers.Input(shape=(X_train_emb.shape[1],)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid")
        ])

        ffnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        ffnn_model.summary()

        ffnn_model.fit(
            X_train_emb, y_train,
            validation_data=(X_val_emb, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )

        y_test_pred_nn = (ffnn_model.predict(X_test_emb).ravel() > 0.5).astype(int)
        print("\n[FFNN] Test Accuracy:", accuracy_score(y_test, y_test_pred_nn))
        print(classification_report(y_test, y_test_pred_nn))

        models_dict["ffnn"] = ffnn_model

    # ----------------------------
    # Save models
    # ----------------------------
    save_all_models(models_dict, save_dir=args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arabic human-vs-AI classifier (PyCharm version)")
    parser.add_argument("--input", type=str, required=True, help="Path to df file (.csv or .parquet)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--run_dl", action="store_true", help="Run deep learning (embeddings + FFNN)")
    parser.add_argument("--epochs", type=int, default=10, help="FFNN epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="FFNN batch size")
    main(parser.parse_args())
