import json
import re
import numpy as np
import pandas as pd

# Centralized Column Definitions
NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?"
]

TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
]

CATEGORICAL_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
    "If this painting was a food, what would be?"
]


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
    return re.findall(r"[a-z']+", str(text).lower())


def text_to_bow_matrix(text_series, vocab):
    X = np.zeros((len(text_series), len(vocab)))
    for i, text in enumerate(text_series):
        for word in tokenize(text):
            if word in vocab:
                X[i, vocab[word]] += 1
    return X


def one_hot(labels, num_classes):
    Y = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


def build_vocab(text_series, max_features=500, min_freq=1):
    counts = {}
    for text in text_series:
        for word in tokenize(text):
            counts[word] = counts.get(word, 0) + 1

    words = []
    for word, count in counts.items():
        if count >= min_freq:
            words.append((word, count))

    words.sort(key=lambda x: (-x[1], x[0]))
    words = words[:max_features]

    vocab = {}
    for i, (word, _) in enumerate(words):
        vocab[word] = i
    return vocab


def preprocess_numeric_column(series):
    # Extracts the first numeric value found in each string
    extracted = series.astype(str).str.extract(r"(\d+(\.\d+)?)")[0]
    return pd.to_numeric(extracted, errors="coerce").fillna(0.0)


def prepare_test_features(test_df, params, include_bias=True):
    # 1. Numeric Preprocessing
    for col in NUMERIC_COLS:
        if col in test_df.columns:
            test_df[col] = preprocess_numeric_column(test_df[col])
    
    X_num = test_df[NUMERIC_COLS].to_numpy(dtype=float)
    mean = params["mean"]
    std = params["std"]
    X_num = (X_num - mean) / std

    # 2. Categorical Preprocessing (One-Hot)
    for col in CATEGORICAL_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("Missing").astype(str)
            
    cat_columns = json.loads(str(params["cat_columns_json"].item()))
    X_cat_df = pd.get_dummies(test_df[CATEGORICAL_COLS], drop_first=False)
    X_cat_df = X_cat_df.reindex(columns=cat_columns, fill_value=0)
    X_cat = X_cat_df.to_numpy(dtype=float)

    # 3. Text Preprocessing (BoW)
    for col in TEXT_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("")
            
    text_parts = []
    for col in TEXT_COLS:
        vocab_key = "vocab__" + col
        if vocab_key in params:
            vocab = json.loads(str(params[vocab_key].item()))
            vocab = {k: int(v) for k, v in vocab.items()}
            X_bow = text_to_bow_matrix(test_df[col], vocab)
            text_parts.append(X_bow)

    X_text = np.hstack(text_parts) if text_parts else np.zeros((len(test_df), 0))
    
    # 4. Final Stacking
    X = np.hstack([X_num, X_cat, X_text])
    
    # Prepend bias term (intercept) if requested
    if include_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X