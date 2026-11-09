"""
Lõi tính toán cho chương trình giao diện (app.py).

Mọi hàm ở đây chỉ dùng NumPy/pandas và lớp RidgeScratch tự cài đặt, trả về các
kết quả TRUNG GIAN (ma trận, trị riêng, hệ số co, lịch sử hàm mất mát...) để giao
diện hiển thị từng bước. Không phụ thuộc Streamlit -> kiểm thử độc lập được.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ridge_scratch import RidgeScratch, ridge_path_svd


# ======================================================================
# 1. Dữ liệu
# ======================================================================
def load_diabetes_df():
    from sklearn.datasets import load_diabetes
    ds = load_diabetes(scaled=False, as_frame=True)
    return ds.data, ds.target.rename("target")


def simulate(n=100, p=10, rho=0.9, sigma=3.0, seed=42):
    """X ~ N(0, Σ), Σ_jk = rho (j≠k), Σ_jj = 1; y = Xβ + nhiễu. β thật: 3, 1.5, 0, 0, 2, 0, ..."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:min(p, 5)] = [3, 1.5, 0, 0, 2][:min(p, 5)]
    cov = np.full((p, p), rho) + (1 - rho) * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    y = X @ beta + rng.normal(0, sigma, n)
    cols = [f"x{j + 1}" for j in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y"), pd.Series(beta, index=cols, name="beta_that")


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j) tính bằng bình phương tối thiểu (NumPy)."""
    Xv = X.to_numpy(float)
    rows = []
    for j, col in enumerate(X.columns):
        target = Xv[:, j]
        others = np.delete(Xv, j, axis=1)
        A = np.column_stack([np.ones(len(Xv)), others])
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        resid = target - A @ coef
        r2 = 1 - resid @ resid / np.sum((target - target.mean()) ** 2)
        rows.append({"Đặc trưng": col, "R²_j": r2, "VIF": np.inf if r2 >= 1 else 1 / (1 - r2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False, ignore_index=True)


def train_test_split_np(n, test_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


# ======================================================================
# 2. Chỉ số đánh giá
# ======================================================================
def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r = y - yhat
    return {"MAE": np.mean(np.abs(r)), "RMSE": np.sqrt(np.mean(r ** 2)),
            "R²": 1 - r @ r / np.sum((y - y.mean()) ** 2)}


# ======================================================================
# 3. Ví dụ tính tay: nhập trực tiếp XᵀX (2×2 hoặc lớn hơn) và Xᵀy
# ======================================================================
def hand_example(XtX, Xty, alpha):
    XtX = np.asarray(XtX, float)
    Xty = np.asarray(Xty, float)
    p = len(Xty)
    A = XtX + alpha * np.eye(p)
    eig0 = np.linalg.eigvalsh(XtX)
    eig1 = np.linalg.eigvalsh(A)
    out = {
        "XtX": XtX, "Xty": Xty, "A": A,
        "det_XtX": np.linalg.det(XtX), "det_A": np.linalg.det(A),
        "eig_XtX": eig0, "eig_A": eig1,
        "cond_XtX": eig0.max() / eig0.min() if eig0.min() > 1e-15 else np.inf,
        "cond_A": eig1.max() / eig1.min() if eig1.min() > 1e-15 else np.inf,
        "A_inv": np.linalg.inv(A) if abs(np.linalg.det(A)) > 1e-15 else None,
    }
    out["beta"] = out["A_inv"] @ Xty if out["A_inv"] is not None else None
    out["beta_ols"] = np.linalg.solve(XtX, Xty) if abs(out["det_XtX"]) > 1e-12 else None
    out["shrink"] = eig0 / (eig0 + alpha) if alpha > 0 or eig0.min() > 0 else None
    out["df"] = float(np.sum(out["shrink"])) if out["shrink"] is not None else None
    return out


def hand_example_table(XtX, Xty, alphas):
    rows = []
    for a in alphas:
        h = hand_example(XtX, Xty, a)
        if h["beta"] is None:
            continue
        rows.append({"α": a, **{f"β{j + 1}": b for j, b in enumerate(h["beta"])},
                     "Tổng β": h["beta"].sum(), "Số điều kiện": h["cond_A"], "df(α)": h["df"]})
    return pd.DataFrame(rows)


# ======================================================================
# 4. Chạy Ridge từng bước trên dữ liệu
# ======================================================================
def run_ridge_steps(X: pd.DataFrame, y: pd.Series, alpha=1.0, solver="cholesky",
                    standardize=True, fit_intercept=True, test_size=0.2, seed=42,
                    lr=None, max_iter=20_000):
    names = list(X.columns)
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    n, p = Xv.shape
    S = {"names": names, "n": n, "p": p}

    # Bước 1 — chia train/test
    tr, te = train_test_split_np(n, test_size, seed)
    S["idx_train"], S["idx_test"] = tr, te
    Xtr, Xte, ytr, yte = Xv[tr], Xv[te], yv[tr], yv[te]

    # Bước 2 — chuẩn hóa (học trên train)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    if standardize:
        Ztr, Zte = (Xtr - mu) / sd_safe, (Xte - mu) / sd_safe
    else:
        Ztr, Zte = Xtr.copy(), Xte.copy()
    S["scaler"] = pd.DataFrame({"Đặc trưng": names, "Trung bình (train)": mu, "Độ lệch chuẩn (train)": sd})
    S["Z_train_head"] = pd.DataFrame(Ztr[:5], columns=names)

    # Bước 3 — căn giữa
    zbar = Ztr.mean(axis=0) if fit_intercept else np.zeros(p)
    ybar = ytr.mean() if fit_intercept else 0.0
    Zc, yc = Ztr - zbar, ytr - ybar
    S["zbar"], S["ybar"] = zbar, ybar

    # Bước 4 — XᵀX, Xᵀy
    XtX = Zc.T @ Zc
    Xty = Zc.T @ yc
    S["XtX"] = pd.DataFrame(XtX, index=names, columns=names)
    S["Xty"] = pd.Series(Xty, index=names, name="Xᵀy")

    # Bước 5 — cộng αI và kiểm tra điều kiện số
    eig0 = np.linalg.eigvalsh(XtX)
    eig1 = eig0 + alpha
    S["eig"] = pd.DataFrame({"Trị riêng XᵀX": eig0[::-1], "Trị riêng XᵀX + αI": eig1[::-1]})
    S["cond_before"] = eig0.max() / eig0.min() if eig0.min() > 1e-12 else np.inf
    S["cond_after"] = eig1.max() / eig1.min() if eig1.min() > 1e-12 else np.inf

    # Bước 6 — giải
    model = RidgeScratch(alpha=alpha, solver=solver, fit_intercept=fit_intercept,
                         lr=lr, max_iter=max_iter, tol=1e-10)
    model.fit(Ztr, ytr)
    detail = {}
    if solver == "cholesky":
        L = np.linalg.cholesky(XtX + alpha * np.eye(p))
        detail["L"] = pd.DataFrame(L, index=names, columns=names)
    elif solver == "svd":
        d = np.linalg.svd(Zc, compute_uv=False)
        detail["svd"] = pd.DataFrame({"Hướng chính j": np.arange(1, len(d) + 1), "d_j": d, "d_j²": d ** 2,
                                      "Hệ số co d²/(d²+α)": d ** 2 / (d ** 2 + alpha)})
        detail["df"] = float(np.sum(d ** 2 / (d ** 2 + alpha)))
    elif solver == "dual":
        detail["K_shape"] = (Zc.shape[0], Zc.shape[0])
    elif solver == "gd":
        detail["loss"] = np.array(model.loss_history_)
        detail["n_iter"] = model.n_iter_
        detail["lr"] = lr if lr is not None else 1 / (2 * (eig0.max() + alpha))
        detail["converged"] = model.n_iter_ < max_iter
    S["solver_detail"] = detail
    S["coef_scaled"] = pd.Series(model.coef_, index=names)
    S["intercept_scaled"] = model.intercept_

    # Bước 7 — đưa hệ số về thang đo gốc
    if standardize:
        coef_o = model.coef_ / sd_safe
        int_o = model.intercept_ - np.sum(model.coef_ * mu / sd_safe)
    else:
        coef_o, int_o = model.coef_.copy(), model.intercept_
    S["coef_original"] = pd.Series(coef_o, index=names)
    S["intercept_original"] = int_o

    # Bước 8 — OLS để đối chiếu (lstsq: vẫn chạy khi suy biến, báo hạng)
    A = np.column_stack([np.ones(len(Ztr)), Ztr]) if fit_intercept else Ztr
    sol, _, rank, _ = np.linalg.lstsq(A, ytr, rcond=None)
    ols_coef = sol[1:] if fit_intercept else sol
    ols_int = sol[0] if fit_intercept else 0.0
    S["ols_rank"], S["ols_full_rank"] = int(rank), int(rank) == A.shape[1]
    S["coef_ols_scaled"] = pd.Series(ols_coef, index=names)

    # Bước 9 — dự đoán và đánh giá
    pred = {"Ridge": (model.predict(Ztr), model.predict(Zte)),
            "OLS": (Ztr @ ols_coef + ols_int, Zte @ ols_coef + ols_int)}
    rows = []
    for name, (ptr, pte) in pred.items():
        for part, yy, pp in [("Train", ytr, ptr), ("Test", yte, pte)]:
            rows.append({"Mô hình": name, "Tập": part, **metrics(yy, pp)})
    S["metrics"] = pd.DataFrame(rows)
    S["pred_test"] = pd.DataFrame({"y thật": yte, "ŷ Ridge": pred["Ridge"][1], "ŷ OLS": pred["OLS"][1]})
    S["coef_norm"] = {"Ridge": float(np.linalg.norm(model.coef_)), "OLS": float(np.linalg.norm(ols_coef))}
    return S


# ======================================================================
# 5. Khảo sát α bằng K-fold CV tự cài đặt (một SVD/fold cho mọi α)
# ======================================================================
def alpha_survey(X: pd.DataFrame, y: pd.Series, alphas, k=5, seed=42, standardize=True,
                 test_size=0.2):
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    tr, _ = train_test_split_np(len(Xv), test_size, seed)       # chỉ dùng phần train
    Xv, yv = Xv[tr], yv[tr]
    n = len(Xv)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    cv_err = np.zeros((k, len(alphas)))
    tr_err = np.zeros((k, len(alphas)))
    for f, val in enumerate(folds):
        fit = np.setdiff1d(np.arange(n), val)
        Xa, Xb, ya, yb = Xv[fit], Xv[val], yv[fit], yv[val]
        if standardize:
            m, s = Xa.mean(0), Xa.std(0)
            s = np.where(s > 0, s, 1)
            Xa, Xb = (Xa - m) / s, (Xb - m) / s
        B = ridge_path_svd(Xa, ya, alphas)                      # (len(alphas), p)
        b0 = ya.mean() - B @ Xa.mean(0)
        cv_err[f] = np.sqrt(np.mean((yb[None, :] - (B @ Xb.T + b0[:, None])) ** 2, axis=1))
        tr_err[f] = np.sqrt(np.mean((ya[None, :] - (B @ Xa.T + b0[:, None])) ** 2, axis=1))

    Z = Xv
    if standardize:
        s = Z.std(0)
        Z = (Z - Z.mean(0)) / np.where(s > 0, s, 1)
    path = ridge_path_svd(Z, yv, alphas)
    d2 = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    res = pd.DataFrame({"alpha": alphas,
                        "RMSE_train": tr_err.mean(0),
                        "RMSE_CV": cv_err.mean(0),
                        "RMSE_CV_std": cv_err.std(0),
                        "df": [np.sum(d2 / (d2 + a)) for a in alphas],
                        "norm_beta": np.linalg.norm(path, axis=1)})
    coef_path = pd.DataFrame(path, columns=X.columns)
    coef_path.insert(0, "alpha", alphas)
    i = int(res["RMSE_CV"].idxmin())
    se = res.loc[i, "RMSE_CV_std"] / np.sqrt(k)
    i1 = int(res.index[res["RMSE_CV"] <= res.loc[i, "RMSE_CV"] + se].max())
    return res, coef_path, float(alphas[i]), float(alphas[i1])


# ======================================================================
# 6. So sánh OLS / Ridge / Lasso / Elastic Net (scikit-learn)
# ======================================================================
def compare_models(X: pd.DataFrame, y: pd.Series, test_size=0.2, seed=42, k=5):
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te = train_test_split_np(len(X), test_size, seed)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    specs = {
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"ridge__alpha": np.logspace(-4, 4, 40)}),
        "Lasso": (Lasso(max_iter=50_000), {"lasso__alpha": np.logspace(-3, 2, 40)}),
        "ElasticNet": (ElasticNet(max_iter=50_000), {"elasticnet__alpha": np.logspace(-3, 2, 20),
                                                     "elasticnet__l1_ratio": [0.1, 0.5, 0.9]}),
    }
    rows, coefs = [], {}
    for name, (est, grid) in specs.items():
        g = GridSearchCV(make_pipeline(StandardScaler(), est), grid,
                         cv=KFold(k, shuffle=True, random_state=seed),
                         scoring="neg_root_mean_squared_error").fit(Xtr, ytr)
        best = g.best_estimator_
        params = ", ".join(f"{k.split('__')[1]}={v:.4g}" for k, v in g.best_params_.items()) or "—"
        mt = metrics(yte, best.predict(Xte))
        rows.append({"Mô hình": name, "Siêu tham số chọn bằng CV": params,
                     "RMSE CV": -g.best_score_, "MAE test": mt["MAE"], "RMSE test": mt["RMSE"],
                     "R² test": mt["R²"], "Số hệ số ≠ 0": int(np.sum(np.abs(best[-1].coef_) > 1e-10))})
        coefs[name] = best[-1].coef_
    return pd.DataFrame(rows), pd.DataFrame(coefs, index=X.columns)
