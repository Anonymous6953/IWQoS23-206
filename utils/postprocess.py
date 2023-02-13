import numpy as np


def abs_anomaly_score(truth_value: np.ndarray, predicted_value: np.ndarray):
    return abs(truth_value - predicted_value)


def static_threshold(anomaly_socres: np.ndarray, threshold):
    predicted_label = np.zeros_like(anomaly_socres)
    predicted_label[anomaly_socres > threshold] = 1

    return predicted_label



