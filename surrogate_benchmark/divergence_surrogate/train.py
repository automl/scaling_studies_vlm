import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score, \
    ConfusionMatrixDisplay, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from surrogate_benchmark.divergence_surrogate.predictors.log_reg import sweep_logreg

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from tabpfn import TabPFNClassifier

    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", default="splits/train.csv")
    p.add_argument("--test_csv", default="splits/test.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        choices=["logreg", "rf", "xgb", "xgb_ens", "tabpfn"],
        default="logreg",
    )

    return p.parse_args()


def build_rf(seed):
    return RandomForestClassifier(
        n_estimators=1000,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        max_depth=8,
        max_features="sqrt",
        random_state=seed)


def build_xgb(seed):
    if not HAS_XGB:
        raise ImportError("xgboost not installed")

    return XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=10,
        eval_metric="aucpr",
        random_state=seed
    )


def xgb_ensemble(seed):
    models = []

    for depth in [3, 5, 9]:
        for n_estimators in [300, 500, 800]:
            for lr in [0.01, 0.05, 0.1, 0.2]:
                models.append(
                    XGBClassifier(
                        max_depth=depth,
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="aucpr",
                        scale_pos_weight=1,
                        random_state=seed)
                )

    return models


def build_tabpfn(seed):
    if not HAS_TABPFN:
        raise ImportError("tabpfn not installed")

    return TabPFNClassifier(device="cuda")


def evaluate_predictions(y_test, y_pred, y_prob):
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])

    tn, fp, fn, tp = cm.ravel()

    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    plt.figure(figsize=(8, 6))
    cmd.plot(cmap='Blues')
    plt.title("Confusion matrix")
    plt.grid(False)
    plt.show()

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nROC AUC:", roc_auc_score(y_test, y_prob))
    print("PR AUC:", average_precision_score(y_test, y_prob))

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def add_engineered_features(df):
    df = df.copy()

    df["lr"] = np.log(df["lr"] + 1e-12)
    df["eps"] = np.log(df["eps"] + 1e-12)
    df["wd"] = np.log(df["wd"] + 1e-12)
    df["total_samples_planned"] = np.log(df["total_samples_planned"] + 1e-12)

    df["lr_batch_ratio"] = df["lr"] / (df["global_batch_size"] + 1e-12)
    df["lr_wd_ratio"] = df["lr"] / (df["wd"] + 1e-12)
    df["width_ratio"] = df["text_width"] / (df["vision_width"] + 1e-12)

    df["lr_batch_product"] = df["lr"] * df["global_batch_size"]
    df["lr_warmup_product"] = df["lr"] * df["warmup_fraction"]
    df["width_product"] = df["text_width"] * df["vision_width"]

    df["total_width"] = df["text_width"] + df["vision_width"]
    df["width_diff"] = (df["text_width"] - df["vision_width"]).abs()

    return df


def oversample_minority(X_train, y_train, seed):
    train_df = X_train.copy()
    train_df["failed"] = y_train.values

    minority_df = train_df[train_df["failed"] == 1]
    majority_df = train_df[train_df["failed"] == 0]

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=seed,
    )

    balanced_df = pd.concat([majority_df, minority_upsampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    X_balanced = balanced_df.drop(columns=["failed"])
    y_balanced = balanced_df["failed"]

    return X_balanced, y_balanced


def evaluate(model, threshold, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n=== TEST RESULTS ===")
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("PR AUC:", average_precision_score(y_test, y_prob))
    print("Threshold:", threshold)

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # train_df = add_engineered_features(train_df)
    # test_df = add_engineered_features(test_df)

    feature_cols = [c for c in train_df.columns if c not in ["config_id", "failed"]]

    X_train = train_df[feature_cols]
    y_train = train_df["failed"]

    X_test = test_df[feature_cols]
    y_test = test_df["failed"]

    # X_train, y_train = oversample_minority(X_train, y_train, args.seed)
    # print("Train class counts after oversampling:")
    # print(y_train.value_counts().sort_index())

    if args.model == "logreg":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model, threshold = sweep_logreg(X_train, y_train, seed=42)
        evaluate(model, threshold, X_test, y_test)

    elif args.model == "rf":
        model = build_rf(args.seed)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    elif args.model == "xgb":
        model = build_xgb(args.seed)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    elif args.model == "xgb_ens":
        models = xgb_ensemble(args.seed)

        for m in models:
            m.fit(X_train, y_train)

        probs = np.zeros(len(X_test))

        for m in models:
            probs += m.predict_proba(X_test)[:, 1]

        probs /= len(models)

        y_prob = probs
        y_pred = (y_prob > 0.5).astype(int)

    elif args.model == "tabpfn":
        model = build_tabpfn(args.seed)

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

    # evaluate_predictions(y_test, y_pred, y_prob)
    # plt.hist(y_prob, bins=30)
    # plt.show()
