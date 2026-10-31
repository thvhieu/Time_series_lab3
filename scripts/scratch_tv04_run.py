from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path.cwd()
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from lab3.data import load_prepared, split_supervised
from lab3.features import make_supervised, xy
from lab3.evaluation import regression_metrics
from lab3.modeling import make_model

cfg = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
frame, quality = load_prepared(ROOT / cfg["prepared_dataset_dir"], raw_path=ROOT / cfg["dataset_path"])
supervised = make_supervised(frame, p=cfg["default_p"],
                             rolling_windows=tuple(cfg["rolling_windows"]), warmup=cfg["warmup_hours"])
train, test = split_supervised(supervised, cfg["test_start"])
X_train, y_train = xy(train)
X_test, y_test = xy(test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# 1. XGBoost cơ sở
print("\n--- 1. Fit XGBoost cơ sở ---")
model = make_model("xgboost", seed=cfg["seed"])
t0 = time.perf_counter()
model.fit(X_train, y_train)
fit_time = time.perf_counter() - t0
print(f"Fit time: {fit_time:.2f}s")

y_train_pred = pd.Series(model.predict(X_train), index=y_train.index, name="y_pred_train")
train_metrics = regression_metrics(y_train, y_train_pred)
print("Train metrics:", train_metrics)

# 2. TimeSeriesSplit CV soát overfit
print("\n--- 2. Soát overfit trên CV (TimeSeriesSplit) ---")
n_splits = cfg["cv_splits"]
test_size = cfg["cv_test_hours"]
tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

cv_records = []
for fold_idx, (tr_idx, val_idx) in enumerate(tss.split(X_train), 1):
    X_tr_f, y_tr_f = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val_f, y_val_f = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    m_f = make_model("xgboost", seed=cfg["seed"])
    t_start = time.perf_counter()
    m_f.fit(X_tr_f, y_tr_f)
    t_f = time.perf_counter() - t_start
    
    pred_tr_f = pd.Series(m_f.predict(X_tr_f), index=y_tr_f.index)
    pred_val_f = pd.Series(m_f.predict(X_val_f), index=y_val_f.index)
    
    met_tr = regression_metrics(y_tr_f, pred_tr_f)
    met_val = regression_metrics(y_val_f, pred_val_f)
    
    cv_records.append({
        "fold": fold_idx,
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
        "val_start": str(y_val_f.index[0]),
        "val_end": str(y_val_f.index[-1]),
        "train_RMSE": round(met_tr["RMSE"], 2),
        "val_RMSE": round(met_val["RMSE"], 2),
        "train_MAE": round(met_tr["MAE"], 2),
        "val_MAE": round(met_val["MAE"], 2),
        "train_MAPE": round(met_tr["MAPE"], 2),
        "val_MAPE": round(met_val["MAPE"], 2),
        "gap_RMSE": round(met_val["RMSE"] - met_tr["RMSE"], 2),
        "fit_seconds": round(t_f, 2)
    })

cv_df = pd.DataFrame(cv_records)
print("CV Overfit Table:")
print(cv_df.to_string(index=False))

# 3. Prediction contract
print("\n--- 3. Prediction Contract on Test ---")
t_pred_start = time.perf_counter()
y_test_pred = pd.Series(model.predict(X_test), index=y_test.index, name="y_pred")
pred_time = time.perf_counter() - t_pred_start
test_metrics = regression_metrics(y_test, y_test_pred)
print("Test metrics:", test_metrics)

# 4. Feature importance
print("\n--- 4. Feature Importance (gain) ---")
importance_gain = model.get_booster().get_score(importance_type="gain")
fi_series = pd.Series(importance_gain).reindex(X_train.columns).fillna(0)
fi_df = pd.DataFrame({
    "feature": fi_series.index,
    "gain": fi_series.values,
    "importance_pct": (fi_series.values / fi_series.sum() * 100).round(2)
}).sort_values(by="gain", ascending=False).reset_index(drop=True)
print("Top 15 features:")
print(fi_df.head(15).to_string(index=False))
