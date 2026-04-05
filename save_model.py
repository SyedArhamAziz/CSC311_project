import json
import numpy as np
import pandas as pd
from model_utils import (
    softmax, tokenize, text_to_bow_matrix, preprocess_numeric_column,
    one_hot, build_vocab, NUMERIC_COLS, TEXT_COLS, CATEGORICAL_COLS
)


def prepare_train_features(train_df, include_bias=True):
    # 1. Numeric Preprocessing
    for col in NUMERIC_COLS:
        if col in train_df.columns:
            train_df[col] = preprocess_numeric_column(train_df[col])

    X_num = train_df[NUMERIC_COLS].to_numpy(dtype=float)
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    std[std == 0] = 1.0
    X_num = (X_num - mean) / std

    # 2. Categorical Preprocessing (One-Hot)
    for col in CATEGORICAL_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("Missing").astype(str)
            
    X_cat_df = pd.get_dummies(train_df[CATEGORICAL_COLS], drop_first=False)
    cat_columns = list(X_cat_df.columns)
    X_cat = X_cat_df.to_numpy(dtype=float)

    # 3. Text Preprocessing (BoW)
    for col in TEXT_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("")
            
    vocab_store = {}
    text_parts = []
    for col in TEXT_COLS:
        vocab = build_vocab(train_df[col], max_features=500, min_freq=1)
        vocab_store[col] = vocab
        X_bow = text_to_bow_matrix(train_df[col], vocab)
        text_parts.append(X_bow)

    X_text = np.hstack(text_parts) if text_parts else np.zeros((len(train_df), 0))
    
    # 4. Final Stacking
    X = np.hstack([X_num, X_cat, X_text])
    
    # Prepend bias term (intercept) if requested
    if include_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, mean, std, cat_columns, vocab_store


def train_model(X, Y, classes, epochs=4000, lr=0.1, reg=0.01,):
    n, d = X.shape
    c = len(classes)
    
    rng = np.random.default_rng(42)
    W = rng.normal(loc=0.0, scale=0.01, size=(d, c))
    
    for epoch in range(epochs):
        P = softmax(X @ W)
        grad_W = (X.T @ (P - Y)) / n + reg * W
        W -= lr * grad_W
        
        if epoch % 500 == 0:
            pred_idx = np.argmax(P, axis=1)
            true_idx = np.argmax(Y, axis=1)
            acc = np.mean(pred_idx == true_idx)
            print(f"Epoch {epoch:4d} | Train Acc: {acc:.4f}")
            
    return W


def main():
    train_df = pd.read_csv("ml_challenge_dataset_fixed.csv")
    train_df = train_df.dropna(subset=["Painting"]).reset_index(drop=True)

    print("Preparing features...")
    X, mean, std, cat_columns, vocab_store = prepare_train_features(train_df.copy(), include_bias=True)

    classes = np.array(sorted(train_df["Painting"].unique()))
    class_to_int = {classes[i]: i for i in range(len(classes))}

    t_train = train_df["Painting"].to_numpy()
    t_train_int = np.array([class_to_int[label] for label in t_train])
    Y_train = one_hot(t_train_int, len(classes))

    print(f"Training on {X.shape[0]} samples with {X.shape[1]} features...")
    W = train_model(X, Y_train, classes)

    save_dict = {
        "W": W,
        "classes": classes,
        "mean": mean,
        "std": std,
        "cat_columns_json": np.array(json.dumps(cat_columns), dtype=object)
    }

    for col, vocab in vocab_store.items():
        save_dict["vocab__" + col] = np.array(json.dumps(vocab), dtype=object)

    np.savez("model_params.npz", **save_dict)
    print("\nSuccessfully saved model_params.npz")


if __name__ == "__main__":
    main()