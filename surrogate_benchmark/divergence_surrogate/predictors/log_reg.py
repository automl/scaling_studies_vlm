import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from surrogate_benchmark.divergence_surrogate.utils import find_best_threshold


def sweep_logreg(X_train, y_train, seed):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=10000,
            random_state=seed
        ))
    ])

    param_dist = {
        "clf__C": np.logspace(-3, 2, 30),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        random_state=seed,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV PR AUC:", search.best_score_)

    oof_prob = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    best_thresh, best_f1 = find_best_threshold(y_train, oof_prob)
    print("Best threshold:", best_thresh)
    print("Best F1 at threshold:", best_f1)

    return best_model, best_thresh
