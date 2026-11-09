"""
TN5 — Dữ liệu MÔ PHỎNG do bài tự sinh (không phải dữ liệu thực).
  (a) Overfitting: đa thức bậc 9 khớp 10 điểm bằng OLS và Ridge (Hình 5.2).
  (b) Đa cộng tuyến: độ ổn định hệ số của OLS và Ridge khi tương quan rho tăng (Hình 5.6).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import common as U
from . import config as C


# ---------------------------------------------------------------- (a)
def true_f(x):
    return np.sin(2 * np.pi * x)


def poly_experiment(seed=C.SEED, n_train=10, degree=9, noise=0.25):
    rng = np.random.default_rng(seed)
    x_tr = np.sort(rng.uniform(0, 1, n_train))
    y_tr = true_f(x_tr) + rng.normal(0, noise, n_train)
    x_te = rng.uniform(0, 1, 1000)
    y_te = true_f(x_te) + rng.normal(0, noise, 1000)
    grid = np.linspace(0, 1, 400)

    # Các alpha được chọn THỦ CÔNG để minh họa (không tinh chỉnh) - ghi rõ trong báo cáo
    settings = [("OLS (α = 0)", None), ("Ridge α = 0,01", 0.01), ("Ridge α = 1", 1.0)]
    rows, curves = [], {}
    for label, a in settings:
        reg = LinearRegression() if a is None else Ridge(alpha=a)
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)
        model.fit(x_tr[:, None], y_tr)
        curves[label] = model.predict(grid[:, None])
        coef = model[-1].coef_
        rows.append({"model": label, "alpha": 0 if a is None else a,
                     "RMSE_train": np.sqrt(np.mean((model.predict(x_tr[:, None]) - y_tr) ** 2)),
                     "RMSE_test": np.sqrt(np.mean((model.predict(x_te[:, None]) - y_te) ** 2)),
                     "max_abs_coef": np.abs(coef).max(),
                     "coef_norm": np.linalg.norm(coef)})
    res = pd.DataFrame(rows)
    U.save_csv(res, "tn5a_polynomial.csv")
    pd.DataFrame({"x": x_tr, "y": y_tr}).pipe(U.save_csv, "tn5a_train_points.csv")
    print(res.round(4).to_string(index=False))

    U.style()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(grid, true_f(grid), color=C.GRAY, linestyle="--", linewidth=1.5, label="Hàm thật sin(2πx)")
    colors = [C.COLORS["OLS"], C.COLORS["Lasso"], C.COLORS["Ridge"]]
    for (label, _), c in zip(settings, colors):
        ax.plot(grid, curves[label], color=c, label=label)
    ax.scatter(x_tr, y_tr, s=40, color=C.TEXT, zorder=5, label="10 điểm huấn luyện")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Đa thức bậc 9 khớp 10 điểm (dữ liệu mô phỏng)")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    U.save(fig, "hinh_5_2_polynomial.png")


# ---------------------------------------------------------------- (b)
BETA_TRUE = np.array([3, 1.5, 0, 0, 2, 0, 0, 0, 0, 0], dtype=float)


def make_data(rng, n, rho, sigma=3.0):
    """X ~ N(0, Σ) với Σ_jj = 1, Σ_jk = rho (tương quan đều giữa mọi cặp)."""
    p = len(BETA_TRUE)
    cov = np.full((p, p), rho) + (1 - rho) * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    y = X @ BETA_TRUE + rng.normal(0, sigma, n)
    return X, y


def collinearity_experiment(n=100, reps=100, rhos=(0.0, 0.5, 0.9, 0.99), seed=C.SEED):
    rng = np.random.default_rng(seed)
    alphas = np.logspace(-3, 3, 61)
    rows, coef_rows = [], []
    for rho in rhos:
        for r in range(reps):
            X, y = make_data(rng, n, rho)
            X_te, y_te = make_data(rng, 1000, rho)
            # X đã có phương sai 1 theo cách sinh -> không cần chuẩn hóa thêm
            ols = LinearRegression().fit(X, y)
            ridge = RidgeCV(alphas=alphas).fit(X, y)       # chọn alpha bằng LOOCV hiệu quả
            for name, m in [("OLS", ols), ("Ridge", ridge)]:
                rmse = np.sqrt(np.mean((m.predict(X_te) - y_te) ** 2))
                rows.append({"rho": rho, "rep": r, "model": name, "RMSE_test": rmse,
                             "coef_sq_error": np.sum((m.coef_ - BETA_TRUE) ** 2),
                             "alpha": getattr(m, "alpha_", 0.0)})
                coef_rows.append({"rho": rho, "rep": r, "model": name,
                                  **{f"b{j + 1}": c for j, c in enumerate(m.coef_)}})
    raw = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)
    U.save_csv(raw, "tn5b_collinearity_raw.csv")
    U.save_csv(coefs, "tn5b_coefficients_raw.csv")

    bcols = [f"b{j + 1}" for j in range(len(BETA_TRUE))]
    summ = []
    for (rho, model), g in coefs.groupby(["rho", "model"]):
        rg = raw[(raw.rho == rho) & (raw.model == model)]
        summ.append({"rho": rho, "model": model,
                     "b1_mean": g["b1"].mean(), "b1_std": g["b1"].std(),
                     "avg_coef_std": g[bcols].std().mean(),
                     "coef_sq_error_mean": rg["coef_sq_error"].mean(),
                     "RMSE_test_mean": rg["RMSE_test"].mean(),
                     "alpha_median": rg["alpha"].median()})
    summ = pd.DataFrame(summ)
    U.save_csv(summ, "tn5b_summary.csv")
    print(summ.round(3).to_string(index=False))

    U.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(rhos))
    w = 0.36
    for k, model in enumerate(["OLS", "Ridge"]):
        s = summ[summ.model == model].sort_values("rho")
        axes[0].bar(x + (k - 0.5) * w, s["b1_std"], width=w - 0.02, color=C.COLORS[model], label=model)
        axes[1].bar(x + (k - 0.5) * w, s["RMSE_test_mean"], width=w - 0.02, color=C.COLORS[model], label=model)
        for xi, v in zip(x + (k - 0.5) * w, s["b1_std"]):
            axes[0].annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=8, color=C.TEXT)
    for ax in axes:
        ax.set_xticks(x, [f"ρ = {r:g}".replace(".", ",") for r in rhos])
        ax.legend()
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Độ lệch chuẩn của β̂₁ qua 100 lần lặp")
    axes[0].set_title("Độ dao động của hệ số β₁ (giá trị thật = 3)", fontsize=10)
    axes[1].set_ylabel("RMSE test trung bình")
    axes[1].set_title("Sai số dự đoán trên dữ liệu mới (σ nhiễu = 3)", fontsize=10)
    axes[1].set_ylim(2.8, None)
    fig.suptitle("OLS và Ridge khi mức đa cộng tuyến tăng (dữ liệu mô phỏng, n = 100, p = 10)",
                 fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.04)
    U.save(fig, "hinh_5_6_collinearity.png")


def main():
    print("[TN5a] Đa thức bậc 9 (mô phỏng)")
    poly_experiment()
    print("[TN5b] Đa cộng tuyến (mô phỏng)")
    collinearity_experiment()


if __name__ == "__main__":
    main()
