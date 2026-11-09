"""
TN1 — So sánh OLS, Ridge, Lasso, ElasticNet trên Diabetes (Bảng 5.3).
Kèm mô tả dữ liệu: thống kê, ma trận tương quan và VIF tính trên TẬP TRAIN.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import common as U
from . import config as C

MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet"]


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j), R²_j: hồi quy cột j theo các cột còn lại (Mục 2.3)."""
    out = []
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        out.append({"feature": col, "R2_j": r2, "VIF": 1 / (1 - r2)})
    return pd.DataFrame(out)


def main():
    print("[TN1] So sánh 4 mô hình trên Diabetes")
    X, y = U.load_data()
    X_tr, X_te, y_tr, y_te = U.split(X, y)
    print(f"  Dữ liệu: {X.shape[0]} mẫu x {X.shape[1]} đặc trưng | train {len(X_tr)}, test {len(X_te)}"
          f" | y trong [{y.min():.0f}, {y.max():.0f}] | thiếu: {int(X.isna().sum().sum())}")

    # --- Mô tả dữ liệu (chỉ dùng tập train để không nhìn vào tập test)
    desc = X_tr.describe().T.reset_index().rename(columns={"index": "feature"})
    U.save_csv(desc, "tn1_data_describe_train.csv")
    corr = X_tr.corr()
    corr.reset_index().rename(columns={"index": "feature"}).pipe(U.save_csv, "tn1_corr_train.csv")
    v = vif(X_tr).sort_values("VIF", ascending=False)
    U.save_csv(v, "tn1_vif_train.csv")
    print("  VIF (train):", ", ".join(f"{r.feature}={r.VIF:.1f}" for r in v.itertuples()))

    # --- Huấn luyện và đánh giá
    rows, coefs = [], {}
    for name in MODELS:
        row, best = U.tune_and_evaluate(name, X_tr, y_tr, X_te, y_te)
        rows.append(row)
        coefs[name] = best[-1].coef_
    res = pd.DataFrame(rows)
    U.save_csv(res, "tn1_metrics.csv")
    coef_df = pd.DataFrame(coefs, index=X.columns).reset_index().rename(columns={"index": "feature"})
    U.save_csv(coef_df, "tn1_coefficients_standardized.csv")

    cols = ["model", "alpha", "l1_ratio", "RMSE_train", "RMSE_CV_mean", "RMSE_CV_std",
            "MAE_test", "RMSE_test", "R2_test", "n_nonzero_coef"]
    print(res.reindex(columns=cols).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
