import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1
