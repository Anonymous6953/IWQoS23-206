import numpy as np
from sklearn.metrics import mean_squared_error
from .evaluate import evaluate_api, mse, LOF
from .postprocess import static_threshold


def best_f1_score(truth_labels, anomaly_scores: np.ndarray, threshold_range=[0, 2], try_cnt=200):
    try_step = (threshold_range[1] - threshold_range[0]) / try_cnt
    try_thresholds = [threshold_range[0] + i * try_step for i in range(0, try_cnt)]

    f1_best = -1

    for ths in try_thresholds:
        predicted_labels = static_threshold(anomaly_scores, ths)

        f1 = evaluate_api(truth_labels, predicted_labels)['data']['f']
        if f1 > f1_best:
            f1_best = f1

    return f1_best


def mse_factor(truth_values: np.ndarray, predicted_values: np.ndarray):
    return mse(truth_values, predicted_values, scale=True)


def normal_factor(predicted_values: np.ndarray):
    return LOF(predicted_values)
