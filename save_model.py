import json
import numpy as np
import pandas as pd
from model_utils import softmax, tokenize, text_to_bow_matrix, preprocess_numeric_column


def one_hot(labels, num_classes):
    Y = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


def build_vocab(text_series, max_features=300, min_freq=2):
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


def prepare_train_features(train_df):
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
        if col in train_df.columns:
            train_df[col] = preprocess_numeric_column(train_df[col])

    for col in text_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("")

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("Missing").astype(str)

    X_num = train_df[numeric_cols].to_numpy(dtype=float)
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    std[std == 0] = 1.0
    X_num = (X_num - mean) / std

    X_cat_df = pd.get_dummies(train_df[categorical_cols], drop_first=False)
    cat_columns = list(X_cat_df.columns)
    X_cat = X_cat_df.to_numpy(dtype=float)

    vocab_store = {}
    text_parts = []
    for col in text_cols:
        vocab = build_vocab(train_df[col], max_features=300, min_freq=2)
        vocab_store[col] = vocab
        X_bow = text_to_bow_matrix(train_df[col], vocab)
        text_parts.append(X_bow)

    X_text = np.hstack(text_parts)
    X = np.hstack([X_num, X_cat, X_text])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, mean, std, cat_columns, vocab_store


def main():
    train_df = pd.read_csv("ml_challenge_dataset_fixed.csv")
    train_df = train_df.dropna(subset=["Painting"]).reset_index(drop=True)

    X, mean, std, cat_columns, vocab_store = prepare_train_features(train_df.copy())

    classes = np.array(sorted(train_df["Painting"].unique()))
    class_to_int = {classes[i]: i for i in range(len(classes))}

    t_train = train_df["Painting"].to_numpy()
    t_train_int = np.array([class_to_int[label] for label in t_train])
    Y_train = one_hot(t_train_int, len(classes))

    n, d = X.shape
    c = len(classes)

    rng = np.random.default_rng(42)
    W = rng.normal(loc=0.0, scale=0.01, size=(d, c))

    learning_rate = 0.05
    num_epochs = 4000
    lambda_reg = 0.001

    for epoch in range(num_epochs):
        P = softmax(X @ W)
        grad_W = (X.T @ (P - Y_train)) / n + lambda_reg * W
        W -= learning_rate * grad_W

        if epoch % 500 == 0:
            pred = classes[np.argmax(P, axis=1)]
            acc = np.mean(pred == t_train)
            print(f"epoch={epoch}, train_acc={acc:.4f}")

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
    print("Saved model_params.npz")


if __name__ == "__main__":
    main()