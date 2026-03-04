import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score


def mean_absolute_percentage_relative_deviation(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = 2 * np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true)

    mask = denominator != 0
    values = np.zeros_like(denominator, dtype=float)
    values[mask] = numerator[mask] / denominator[mask]

    return np.mean(values) * 100


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    marpd = mean_absolute_percentage_relative_deviation(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = pearsonr(y_true, y_pred).statistic
    spearman_corr = spearmanr(y_true, y_pred).correlation

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mdae": float(mdae),
        "marpd": float(marpd),
        "r2": float(r2),
        "r": float(r),
        "spearman": float(spearman_corr),
    }
