"""
TN0 — Kiểm chứng cài đặt tự viết.
  (a) Tái lập Bảng 3.1 (ví dụ tính tay 2x2, Mục 3.3) và hệ số co / df (Mục 3.5).
  (b) So sánh RidgeScratch (4 solver) với sklearn.linear_model.Ridge trên Diabetes.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import common as U
from .ridge_scratch import RidgeScratch, ridge_from_gram


def hand_example():
    XtX = np.array([[1.0, 0.99], [0.99, 1.0]])
    samples = {"A": np.array([1.0, 0.98]), "B": np.array([0.98, 1.0])}
    rows = []
    for alpha in [0.0, 0.1, 1.0]:
        M = XtX + alpha * np.eye(2)
        eig = np.linalg.eigvalsh(M)
        bA = ridge_from_gram(XtX, samples["A"], alpha)
        bB = ridge_from_gram(XtX, samples["B"], alpha)
        mu = np.linalg.eigvalsh(XtX)                 # 0.01 và 1.99
        shrink = mu / (mu + alpha)
        rows.append({
            "alpha": alpha,
            "beta1_A": bA[0], "beta2_A": bA[1],
            "beta1_B": bB[0], "beta2_B": bB[1],
            "det": np.linalg.det(M),
            "eig_min": eig[0], "eig_max": eig[1],
            "cond": eig[1] / eig[0],
            "sum_beta_A": bA.sum(),
            "max_abs_change_A_to_B": np.abs(bA - bB).max(),
            "shrink_dir_diff(mu=0.01)": shrink[0],
            "shrink_dir_common(mu=1.99)": shrink[1],
            "df": shrink.sum(),
        })
    return pd.DataFrame(rows)


def compare_with_sklearn():
    X, y = U.load_data()
    X_tr, _, y_tr, _ = U.split(X, y)
    Xs = StandardScaler().fit_transform(X_tr)
    y_tr = y_tr.to_numpy()
    rows = []
    for alpha in [0.01, 1.0, 10.0, 100.0]:
        ref = Ridge(alpha=alpha).fit(Xs, y_tr)
        for solver in RidgeScratch.SOLVERS:
            m = RidgeScratch(alpha=alpha, solver=solver).fit(Xs, y_tr)
            rows.append({
                "alpha": alpha, "solver": solver,
                "max_abs_diff_coef": np.abs(m.coef_ - ref.coef_).max(),
                "abs_diff_intercept": abs(m.intercept_ - ref.intercept_),
                "R2_train_scratch": m.score(Xs, y_tr),
                "R2_train_sklearn": ref.score(Xs, y_tr),
                "gd_iterations": getattr(m, "n_iter_", np.nan),
            })
    return pd.DataFrame(rows)


def main():
    print("[TN0] Kiểm chứng cài đặt tự viết")
    hand = hand_example()
    U.save_csv(hand, "tn0_hand_example.csv")
    print(hand[["alpha", "beta1_A", "beta2_A", "beta1_B", "beta2_B", "cond", "sum_beta_A", "df"]]
          .round(4).to_string(index=False))
    cmp = compare_with_sklearn()
    U.save_csv(cmp, "tn0_scratch_vs_sklearn.csv")
    worst = cmp["max_abs_diff_coef"].max()
    print(f"  Sai lệch hệ số lớn nhất so với sklearn: {worst:.2e}")
    assert worst < 1e-4, "Cài đặt tự viết lệch khỏi sklearn!"


if __name__ == "__main__":
    main()
