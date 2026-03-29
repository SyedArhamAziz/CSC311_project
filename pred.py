import numpy as np
import pandas as pd
from model_utils import prepare_test_features, pred_multiclass


def predict_all(filename):
    params = np.load("model_params.npz", allow_pickle=True)
    test_df = pd.read_csv(filename)

    X = prepare_test_features(test_df.copy(), params)
    y = pred_multiclass(params["W"], X)

    classes = params["classes"]
    preds = classes[y]
    return list(preds)