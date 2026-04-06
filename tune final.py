
import pandas as pd
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model_utils import (
    one_hot,
    softmax,
    NUMERIC_COLS,
    TEXT_COLS,
    CATEGORICAL_COLS,
    preprocess_numeric_column,
    text_to_bow_matrix,
    build_vocab
)


NON_BOW_RESULTS_FILE = "random_search_results.csv"
BOW_RESULTS_FILE = "bow_search_results.csv"
TOP_K = 5


def train_model(X, Y, epochs, lr, reg):
    n, d = X.shape
    c = Y.shape[1]

    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.01, size=(d, c))

    for _ in range(epochs):
        P = softmax(X @ W)
        grad = (X.T @ (P - Y)) / n + reg * W
        W = W - lr * grad

    return W


def predict_model(X, W, classes):
    probs = softmax(X @ W)
    pred_idx = np.argmax(probs, axis=1)
    return classes[pred_idx]


def prepare_train_features_custom(train_df, max_features, min_freq, include_bias=True):
    train_df = train_df.copy()

    for col in NUMERIC_COLS:
        train_df[col] = preprocess_numeric_column(train_df[col])

    X_num = train_df[NUMERIC_COLS].to_numpy(dtype=float)
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    std[std == 0] = 1.0
    X_num = (X_num - mean) / std

    for col in CATEGORICAL_COLS:
        train_df[col] = train_df[col].fillna("Missing").astype(str)

    X_cat_df = pd.get_dummies(train_df[CATEGORICAL_COLS], drop_first=False)
    cat_columns = list(X_cat_df.columns)
    X_cat = X_cat_df.to_numpy(dtype=float)

    for col in TEXT_COLS:
        train_df[col] = train_df[col].fillna("")

    vocab_store = {}
    text_parts = []

    for col in TEXT_COLS:
        vocab = build_vocab(train_df[col], max_features=max_features, min_freq=min_freq)
        vocab_store[col] = vocab
        X_bow = text_to_bow_matrix(train_df[col], vocab)
        text_parts.append(X_bow)

    if len(text_parts) > 0:
        X_text = np.hstack(text_parts)
    else:
        X_text = np.zeros((len(train_df), 0))

    X = np.hstack([X_num, X_cat, X_text])

    if include_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mean, std, cat_columns, vocab_store


def prepare_validation_features(val_df, mean, std, cat_columns, vocab_store, include_bias=True):
    val_df = val_df.copy()

    for col in NUMERIC_COLS:
        val_df[col] = preprocess_numeric_column(val_df[col])

    X_num = val_df[NUMERIC_COLS].to_numpy(dtype=float)
    X_num = (X_num - mean) / std

    for col in CATEGORICAL_COLS:
        val_df[col] = val_df[col].fillna("Missing").astype(str)

    X_cat_df = pd.get_dummies(val_df[CATEGORICAL_COLS], drop_first=False)
    X_cat_df = X_cat_df.reindex(columns=cat_columns, fill_value=0)
    X_cat = X_cat_df.to_numpy(dtype=float)

    for col in TEXT_COLS:
        val_df[col] = val_df[col].fillna("")

    text_parts = []
    for col in TEXT_COLS:
        vocab = vocab_store[col]
        X_bow = text_to_bow_matrix(val_df[col], vocab)
        text_parts.append(X_bow)

    if len(text_parts) > 0:
        X_text = np.hstack(text_parts)
    else:
        X_text = np.zeros((len(val_df), 0))

    X = np.hstack([X_num, X_cat, X_text])

    if include_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def get_top_values(results_df, col_name, top_k):
    values = results_df[col_name].dropna().tolist()
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
        if len(unique_values) == top_k:
            break
    return unique_values


def main():
    non_bow_results = pd.read_csv(NON_BOW_RESULTS_FILE).sort_values("val_accuracy", ascending=False)
    bow_results = pd.read_csv(BOW_RESULTS_FILE).sort_values("val_accuracy", ascending=False)

    top_epochs = get_top_values(non_bow_results, "epochs", TOP_K)
    top_lr = get_top_values(non_bow_results, "lr", TOP_K)
    top_reg = get_top_values(non_bow_results, "reg", TOP_K)
    top_max_features = get_top_values(bow_results, "max_features", TOP_K)
    top_min_freq = get_top_values(bow_results, "min_freq", TOP_K)

    all_combos = list(product(top_epochs, top_lr, top_reg, top_max_features, top_min_freq))
    print(f"Testing {len(all_combos)} final combinations")

    df = pd.read_csv("ml_challenge_dataset_fixed.csv")
    df = df.dropna(subset=["Painting"]).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Painting"]
    )

    classes = np.array(sorted(train_df["Painting"].unique()))
    class_to_int = {classes[i]: i for i in range(len(classes))}

    y_train = train_df["Painting"].to_numpy()
    y_val = val_df["Painting"].to_numpy()
    y_train_int = np.array([class_to_int[label] for label in y_train])
    Y_train = one_hot(y_train_int, len(classes))

    best_score = -1
    best_params = None
    all_results = []

    for epochs, lr, reg, max_features, min_freq in all_combos:
        epochs = int(epochs)
        lr = float(lr)
        reg = float(reg)
        max_features = int(max_features)
        min_freq = int(min_freq)

        X_train, mean, std, cat_columns, vocab_store = prepare_train_features_custom(
            train_df,
            max_features=max_features,
            min_freq=min_freq,
            include_bias=True
        )

        X_val = prepare_validation_features(
            val_df,
            mean,
            std,
            cat_columns,
            vocab_store,
            include_bias=True
        )

        W = train_model(X_train, Y_train, epochs=epochs, lr=lr, reg=reg)
        preds = predict_model(X_val, W, classes)
        acc = accuracy_score(y_val, preds)

        result = {
            "epochs": epochs,
            "lr": lr,
            "reg": reg,
            "max_features": max_features,
            "min_freq": min_freq,
            "val_accuracy": acc
        }
        all_results.append(result)
        print(result)

        if acc > best_score:
            best_score = acc
            best_params = result.copy()

    results_df = pd.DataFrame(all_results).sort_values("val_accuracy", ascending=False)
    results_df.to_csv("final_combined_search_results.csv", index=False)

    print("\nBest final combination:")
    print(best_params)
    print("Best validation accuracy:")
    print(best_score)


if __name__ == "__main__":
    main()
