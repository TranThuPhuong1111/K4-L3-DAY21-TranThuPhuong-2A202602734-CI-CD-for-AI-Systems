import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu va do lech toi da cho phep (5 diem phan tram)
REFERENCE_POSITIVE_RATE = 0.248
MAX_DRIFT = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: canh bao lech lac du lieu truoc khi huan luyen
    positive_rate = float(y_train.mean())
    drift = abs(positive_rate - REFERENCE_POSITIVE_RATE)
    print(f"Ty le lop duong trong tap huan luyen: {positive_rate:.4f} ({len(df_train)} mau)")
    if drift > MAX_DRIFT:
        print(
            f"::warning::DATA DRIFT - ty le lop duong {positive_rate:.1%} lech "
            f"{drift:.1%} so voi tham chieu {REFERENCE_POSITIVE_RATE:.1%} (nguong {MAX_DRIFT:.0%})"
        )

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_param("n_train_samples", len(df_train))

        # Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout va tinh chi so.
        # f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: quet nguong quyet dinh tu 0.1 den 0.9 (buoc 0.05)
        proba = model.predict_proba(X_eval)[:, 1]
        best_threshold, best_f1 = 0.5, f1
        for t in np.round(np.arange(0.10, 0.9001, 0.05), 2):
            f1_t = float(f1_score(y_eval, (proba >= t).astype(int), zero_division=0))
            if f1_t > best_f1:
                best_threshold, best_f1 = float(t), f1_t

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_f1)
        mlflow.log_metric("train_positive_rate", positive_rate)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Nguong toi uu: {best_threshold:.2f} -> F1 {best_f1:.4f} (nguong 0.5: F1 {f1:.4f})")

        # Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "best_threshold": best_threshold,
                    "best_threshold_f1": best_f1,
                    "train_positive_rate": positive_rate,
                    "n_train_samples": len(df_train),
                },
                f,
                indent=2,
            )

        # Bonus 3: confusion matrix va precision / recall tung lop
        cm = confusion_matrix(y_eval, preds, labels=[0, 1])
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("Confusion matrix (hang = thuc te, cot = du doan)\n")
            f.write("                 pred_thap  pred_cao\n")
            f.write(f"thuc_te_thap     {cm[0, 0]:>9}  {cm[0, 1]:>8}\n")
            f.write(f"thuc_te_cao      {cm[1, 0]:>9}  {cm[1, 1]:>8}\n\n")
            f.write(
                classification_report(
                    y_eval,
                    preds,
                    labels=[0, 1],
                    target_names=["thu_nhap_thap", "thu_nhap_cao"],
                    digits=4,
                    zero_division=0,
                )
            )

        # Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
