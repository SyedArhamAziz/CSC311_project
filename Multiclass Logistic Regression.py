import re
import numpy as np
import pandas as pd

SEED = 42

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def one_hot(labels, num_classes):
    Y = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


def tokenize(text):
    return re.findall(r"[a-z']+", str(text).lower())


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


def text_to_bow_matrix(text_series, vocab):
    X = np.zeros((len(text_series), len(vocab)))
    for i, text in enumerate(text_series):
        for word in tokenize(text):
            if word in vocab:
                X[i, vocab[word]] += 1
    return X


def predict_probs(X, W, b):
    return softmax(X @ W + b)


def preprocess_numeric_column(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+(\.\d+)?)")[0],
        errors="coerce"
    ).fillna(0.0)


def prepare_features(train_df, test_df, max_features=300, min_freq=2):
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
        if col in test_df.columns:
            test_df[col] = preprocess_numeric_column(test_df[col])

    for col in text_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("")
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("")

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("Missing").astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("Missing").astype(str)

    numeric_cols_present = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
    categorical_cols_present = [col for col in categorical_cols if col in train_df.columns and col in test_df.columns]
    text_cols_present = [col for col in text_cols if col in train_df.columns and col in test_df.columns]

    X_train_num = train_df[numeric_cols_present].to_numpy(dtype=float)
    X_test_num = test_df[numeric_cols_present].to_numpy(dtype=float)

    mean = X_train_num.mean(axis=0)
    std = X_train_num.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_num = (X_train_num - mean) / std
    X_test_num = (X_test_num - mean) / std

    if len(categorical_cols_present) > 0:
        X_train_cat_df = pd.get_dummies(train_df[categorical_cols_present], drop_first=False)
        X_test_cat_df = pd.get_dummies(test_df[categorical_cols_present], drop_first=False)

        train_cat_columns = X_train_cat_df.columns
        X_test_cat_df = X_test_cat_df.reindex(columns=train_cat_columns, fill_value=0)

        X_train_cat = X_train_cat_df.to_numpy(dtype=float)
        X_test_cat = X_test_cat_df.to_numpy(dtype=float)
    else:
        X_train_cat = np.zeros((len(train_df), 0))
        X_test_cat = np.zeros((len(test_df), 0))
    X_train_text_parts = []
    X_test_text_parts = []

    for col in text_cols_present:
        vocab = build_vocab(train_df[col], max_features=max_features, min_freq=min_freq)
        X_train_bow = text_to_bow_matrix(train_df[col], vocab)
        X_test_bow = text_to_bow_matrix(test_df[col], vocab)
        X_train_text_parts.append(X_train_bow)
        X_test_text_parts.append(X_test_bow)

    if len(X_train_text_parts) > 0:
        X_train_text = np.hstack(X_train_text_parts)
        X_test_text = np.hstack(X_test_text_parts)
    else:
        X_train_text = np.zeros((len(train_df), 0))
        X_test_text = np.zeros((len(test_df), 0))

    X_train = np.hstack([X_train_num, X_train_cat, X_train_text])
    X_test = np.hstack([X_test_num, X_test_cat, X_test_text])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, X_test


def train_model_with_params(train_df, learning_rate=0.05, num_epochs=4000,
                            lambda_reg=0.001, max_features=300, min_freq=2):
    target_col = "Painting"
    X_train, _ = prepare_features(
        train_df.copy(),
        train_df.copy(),
        max_features=max_features,
        min_freq=min_freq
    )

    classes = np.array(sorted(train_df[target_col].unique()))
    class_to_int = {classes[i]: i for i in range(len(classes))}

    t_train = train_df[target_col].to_numpy()
    t_train_int = np.array([class_to_int[label] for label in t_train])
    Y_train = one_hot(t_train_int, len(classes))

    n, d = X_train.shape
    c = len(classes)

    rng = np.random.default_rng(42)
    W = rng.normal(loc=0.0, scale=0.01, size=(d, c))
    b = np.zeros((1, c))

    for _ in range(num_epochs):
        P = predict_probs(X_train, W, b)

        reg_matrix = lambda_reg * W
        grad_W = (X_train.T @ (P - Y_train)) / n + reg_matrix
        grad_b = np.sum(P - Y_train, axis=0, keepdims=True) / n

        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

    return W, b, classes


def compute_accuracy(df_train, df_valid, learning_rate=0.05, num_epochs=4000,
                     lambda_reg=0.001, max_features=300, min_freq=2):
    W, b, classes = train_model_with_params(
        df_train,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        lambda_reg=lambda_reg,
        max_features=max_features,
        min_freq=min_freq
    )

    _, X_valid = prepare_features(
        df_train.copy(),
        df_valid.copy(),
        max_features=max_features,
        min_freq=min_freq
    )

    t_valid = df_valid["Painting"].to_numpy()
    probs = predict_probs(X_valid, W, b)
    pred_int = np.argmax(probs, axis=1)
    preds = classes[pred_int]

    return np.mean(preds == t_valid)


def evaluate_hyperparameters(df):
    df_train = df.sample(frac=0.8, random_state=SEED)
    df_valid = df.drop(df_train.index).sample(frac=1, random_state=SEED)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    learning_rates = [0.01, 0.03, 0.05, 0.08]
    epoch_values = [3000, 4000, 5000, 7000 ,8000]
    reg_values = [0.0005, 0.001, 0.01]
    max_feature_values = [300, 500, 700, 800]
    min_freq_values = [1, 2, 4, 5]

    results = []

    for lr in learning_rates:
        for epochs in epoch_values:
            for reg in reg_values:
                for max_features in max_feature_values:
                    for min_freq in min_freq_values:
                        acc = compute_accuracy(
                            df_train,
                            df_valid,
                            learning_rate=lr,
                            num_epochs=epochs,
                            lambda_reg=reg,
                            max_features=max_features,
                            min_freq=min_freq
                        )

                        result = {
                            "learning_rate": lr,
                            "num_epochs": epochs,
                            "lambda_reg": reg,
                            "max_features": max_features,
                            "min_freq": min_freq,
                            "valid_accuracy": acc
                        }
                        results.append(result)

                        print(
                            f"lr={lr}, epochs={epochs}, reg={reg}, "
                            f"max_features={max_features}, min_freq={min_freq} "
                            f"-> valid_acc={acc:.4f}"
                        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="valid_accuracy", ascending=False)

    print("\nTop hyperparameter settings:")
    print(results_df.head(10).to_string(index=False))

    best_row = results_df.iloc[0]
    print("\nBest hyperparameters found:")
    print(best_row.to_string())

    return results_df


def train_model(train_df):
    W, b, classes = train_model_with_params(
        train_df,
        learning_rate=0.05,
        num_epochs=4000,
        lambda_reg=0.001,
        max_features=300,
        min_freq=2
    )
    return W, b, classes


def predict_all(filename):
    train_filename = "ml_challenge_dataset_fixed.csv"

    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(filename)

    train_df = train_df.dropna(subset=["Painting"])

    W, b, classes = train_model(train_df.copy())
    _, X_test = prepare_features(train_df.copy(), test_df.copy(), max_features=300, min_freq=2)

    probs = predict_probs(X_test, W, b)
    pred_int = np.argmax(probs, axis=1)
    preds = classes[pred_int]

    return list(preds)


if __name__ == "__main__":
    df = pd.read_csv("ml_challenge_dataset_fixed.csv")
    df = df.dropna(subset=["Painting"]).reset_index(drop=True)

    results_df = evaluate_hyperparameters(df)
    results_df.to_csv("hyperparameter_results.csv", index=False)
    print("\nSaved all hyperparameter results to hyperparameter_results.csv")