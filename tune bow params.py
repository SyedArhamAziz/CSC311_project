
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, ParameterSampler, ParameterGrid
from sklearn.metrics import accuracy_score
from scipy.stats import randint

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


SEARCH_TYPE = "random"   # change to "grid" if you want grid search
N_ITER = 15              # only used for random search

# fixed non-bow hyperparameters from the earlier search
FIXED_EPOCHS = 1000
FIXED_LR = 0.05
FIXED_REG = 0.0


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


def main():
    df = pd.read_csv("ml_challenge_dataset_fixed.csv")
    df = df.dropna(subset=["Painting"]).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Painting"]
    )

    grid_params = {
        "max_features": [100, 250, 500, 750, 1000],
        "min_freq": [1, 2, 3, 5]
    }

    random_params = {
        "max_features": randint(100, 1001),
        "min_freq": randint(1, 6)
    }

    if SEARCH_TYPE == "random":
        param_list = list(ParameterSampler(random_params, n_iter=N_ITER, random_state=42))
        print(f"Using random search with {len(param_list)} sampled BoW combinations")
    elif SEARCH_TYPE == "grid":
        param_list = list(ParameterGrid(grid_params))
        print(f"Using grid search with {len(param_list)} BoW combinations")
    else:
        raise ValueError('SEARCH_TYPE must be "random" or "grid"')

    classes = np.array(sorted(train_df["Painting"].unique()))
    class_to_int = {classes[i]: i for i in range(len(classes))}

    y_train = train_df["Painting"].to_numpy()
    y_val = val_df["Painting"].to_numpy()
    y_train_int = np.array([class_to_int[label] for label in y_train])
    Y_train = one_hot(y_train_int, len(classes))

    best_score = -1
    best_params = None
    all_results = []

    for params in param_list:
        max_features = int(params["max_features"])
        min_freq = int(params["min_freq"])

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

        W = train_model(
            X_train,
            Y_train,
            epochs=FIXED_EPOCHS,
            lr=FIXED_LR,
            reg=FIXED_REG
        )

        preds = predict_model(X_val, W, classes)
        acc = accuracy_score(y_val, preds)

        result = {
            "search_type": SEARCH_TYPE,
            "epochs": FIXED_EPOCHS,
            "lr": FIXED_LR,
            "reg": FIXED_REG,
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
    results_df.to_csv("bow_search_results.csv", index=False)

    print("\nBest BoW hyperparameters:")
    print(best_params)
    print("Best validation accuracy:")
    print(best_score)


if __name__ == "__main__":
    main()
