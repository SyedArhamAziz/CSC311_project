import json
import numpy as np
import pandas as pd


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def pred_probs(W, X):
    return softmax(X @ W)


def pred_multiclass(W, X):
    probs = pred_probs(W, X)
    return np.argmax(probs, axis=1)


def tokenize(text):
    text = str(text).lower()
    cleaned = []
    for ch in text:
        if ("a" <= ch <= "z") or ch == "'":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return "".join(cleaned).split()


def text_to_bow_matrix(text_series, vocab):
    X = np.zeros((len(text_series), len(vocab)))
    for i, text in enumerate(text_series):
        for word in tokenize(text):
            if word in vocab:
                X[i, vocab[word]] += 1
    return X


def preprocess_numeric_column(series):
    vals = []
    for x in series.astype(str):
        num = ""
        seen_digit = False
        seen_dot = False
        for ch in x:
            if ch.isdigit():
                num += ch
                seen_digit = True
            elif ch == "." and seen_digit and not seen_dot:
                num += ch
                seen_dot = True
            elif seen_digit:
                break
        if num == "":
            vals.append(np.nan)
        else:
            vals.append(float(num))
    return pd.Series(vals).fillna(0.0)


def prepare_test_features(test_df, params):
    numeric_cols = [
        "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy.",
        "How many prominent colours do you notice in this painting?",
        "How many objects caught your eye in the painting?",
        "How much (in Canadian dollars) would you be willing to pay for this painting?"
    ]

    text_cols = [
        "Describe how this painting makes you feel.",
        "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
    ]

    categorical_cols = [
        "If you could purchase this painting, which room would you put that painting in?",
        "If you could view this art in person, who would you want to view it with?",
        "What season does this art piece remind you of?",
        "If this painting was a food, what would be?"
    ]

    for col in numeric_cols:
        if col in test_df.columns:
            test_df[col] = preprocess_numeric_column(test_df[col])

    for col in text_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("")

    for col in categorical_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("Missing").astype(str)

    X_num = test_df[numeric_cols].to_numpy(dtype=float)
    mean = params["mean"]
    std = params["std"]
    X_num = (X_num - mean) / std

    cat_columns = json.loads(str(params["cat_columns_json"].item()))
    X_cat_df = pd.get_dummies(test_df[categorical_cols], drop_first=False)
    X_cat_df = X_cat_df.reindex(columns=cat_columns, fill_value=0)
    X_cat = X_cat_df.to_numpy(dtype=float)

    text_parts = []
    for col in text_cols:
        vocab = json.loads(str(params["vocab__" + col].item()))
        vocab = {k: int(v) for k, v in vocab.items()}
        X_bow = text_to_bow_matrix(test_df[col], vocab)
        text_parts.append(X_bow)

    X_text = np.hstack(text_parts)
    X = np.hstack([X_num, X_cat, X_text])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X