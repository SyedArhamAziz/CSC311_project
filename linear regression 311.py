import numpy as np
import pandas as pd


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def one_hot(labels, num_classes):
    Y = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


def cross_entropy_loss(Y, P):
    eps = 1e-12
    P = np.clip(P, eps, 1 - eps)
    return -np.mean(np.sum(Y * np.log(P), axis=1))


def predict_probs(X, W, b):
    scores = X @ W + b
    return softmax(scores)


def predict(X, W, b, classes):
    probs = predict_probs(X, W, b)
    pred_int = np.argmax(probs, axis=1)
    return classes[pred_int]


def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)



df = pd.read_csv("ml_challenge_dataset_fixed.csv")

df = df.dropna(subset=["This art piece makes me feel sombre."])

df = df.drop(columns=[
    "unique_id",
    "Describe how this painting makes you feel.",
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
])

emotion_cols = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]

for col in emotion_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

feature_cols = [col for col in df.columns if col != "Painting"]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[feature_cols] = df[feature_cols].fillna(0)


df_train = df.sample(frac=0.8, random_state=42)
df_valid = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

X_train = df_train.drop(columns=["Painting"]).to_numpy(dtype=float)
X_valid = df_valid.drop(columns=["Painting"]).to_numpy(dtype=float)

t_train = df_train["Painting"].to_numpy()
t_valid = df_valid["Painting"].to_numpy()

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)


classes = np.array(sorted(df["Painting"].unique()))
class_to_int = {classes[i]: i for i in range(len(classes))}

t_train_int = np.array([class_to_int[label] for label in t_train])
t_valid_int = np.array([class_to_int[label] for label in t_valid])

Y_train = one_hot(t_train_int, len(classes))
Y_valid = one_hot(t_valid_int, len(classes))


n, d = X_train.shape
c = len(classes)

W = np.zeros((d, c))
b = np.zeros((1, c))

learning_rate = 0.03
num_epochs = 3000


for epoch in range(num_epochs):
    P = predict_probs(X_train, W, b)

    grad_W = (X_train.T @ (P - Y_train)) / n
    grad_b = np.sum(P - Y_train, axis=0, keepdims=True) / n

    W -= learning_rate * grad_W
    b -= learning_rate * grad_b

    if epoch % 500 == 0:
        train_loss = cross_entropy_loss(Y_train, P)
        print(f"epoch={epoch}, loss={train_loss:.4f}")


train_pred = predict(X_train, W, b, classes)
valid_pred = predict(X_valid, W, b, classes)

train_acc = compute_accuracy(train_pred, t_train)
valid_acc = compute_accuracy(valid_pred, t_valid)

print("\nTrain accuracy:", train_acc)
print("Validation accuracy:", valid_acc)

print("\nPredicted label counts:")
print(pd.Series(valid_pred).value_counts())

print("\nTrue label counts:")
print(pd.Series(t_valid).value_counts())