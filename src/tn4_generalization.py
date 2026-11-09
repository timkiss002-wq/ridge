"""
TN4 — Khả năng tổng quát hóa: lặp lại toàn bộ quy trình TN1 với 20 cách chia dữ liệu
(seed 0..19) để xem độ dao động của sai số test và của alpha được chọn (Hình 5.5).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import common as U
from . import config as C

MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet"]


def main(plot_only=False):
    if plot_only:                         # vẽ lại hình từ CSV đã lưu, không chạy lại
        return plot(pd.read_csv(C.RESULTS / "tn4_seeds_raw.csv"))
    print(f"[TN4] Lặp lại quy trình với {C.N_SEEDS_TN4} seed (mất vài phút)")
    X, y = U.load_data()
    rows = []
    for seed in range(C.N_SEEDS_TN4):
        X_tr, X_te, y_tr, y_te = U.split(X, y, seed=seed)
        for name in MODELS:
            row, _ = U.tune_and_evaluate(name, X_tr, y_tr, X_te, y_te, seed=seed)
            row["seed"] = seed
            rows.append(row)
        print(f"  seed {seed:2d} xong")
    res = pd.DataFrame(rows)
    U.save_csv(res, "tn4_seeds_raw.csv")

    # Chênh lệch có ghép cặp so với OLS trên CÙNG cách chia
    wide = res.pivot(index="seed", columns="model", values="RMSE_test")
    diff = wide.sub(wide["OLS"], axis=0)
    summary = pd.DataFrame({
        "model": MODELS,
        "RMSE_test_mean": [wide[m].mean() for m in MODELS],
        "RMSE_test_std": [wide[m].std() for m in MODELS],
        "R2_test_mean": [res[res.model == m]["R2_test"].mean() for m in MODELS],
        "diff_vs_OLS_mean": [diff[m].mean() for m in MODELS],
        "diff_vs_OLS_std": [diff[m].std() for m in MODELS],
        "seeds_better_than_OLS": [int((diff[m] < 0).sum()) for m in MODELS],
        "alpha_median": [res[res.model == m]["alpha"].median() if m != "OLS" else np.nan for m in MODELS],
        "alpha_min": [res[res.model == m]["alpha"].min() if m != "OLS" else np.nan for m in MODELS],
        "alpha_max": [res[res.model == m]["alpha"].max() if m != "OLS" else np.nan for m in MODELS],
    })
    U.save_csv(summary, "tn4_summary.csv")
    print(summary.round(4).to_string(index=False))
    plot(res)


def plot(res):
    wide = res.pivot(index="seed", columns="model", values="RMSE_test")
    diff = wide.sub(wide["OLS"], axis=0)
    U.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rng = np.random.default_rng(0)
    for k, m in enumerate(MODELS):
        jit = rng.uniform(-0.12, 0.12, len(wide))
        axes[0].scatter(np.full(len(wide), k) + jit, wide[m], s=22, color=C.COLORS[m], alpha=0.8)
        axes[0].plot([k - 0.25, k + 0.25], [wide[m].mean()] * 2, color=C.TEXT, linewidth=2)
        if m != "OLS":
            axes[1].scatter(np.full(len(diff), k - 1) + jit, diff[m], s=22, color=C.COLORS[m], alpha=0.8)
            axes[1].plot([k - 1.25, k - 0.75], [diff[m].mean()] * 2, color=C.TEXT, linewidth=2)
    axes[0].set_xticks(range(4), MODELS)
    axes[0].set_ylabel("RMSE trên tập test")
    axes[0].set_title(f"RMSE test qua {len(wide)} cách chia (gạch đen = trung bình)", fontsize=10)
    axes[1].axhline(0, color=C.TEXT2, linewidth=1)
    axes[1].set_xticks(range(3), MODELS[1:])
    axes[1].set_ylabel("RMSE mô hình − RMSE OLS (cùng cách chia)")
    axes[1].set_title("Chênh lệch ghép cặp với OLS (< 0: tốt hơn OLS)", fontsize=10)
    fig.suptitle("Độ dao động của sai số test theo cách chia dữ liệu",
                 fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.04)
    U.save(fig, "hinh_5_5_seeds.png")


if __name__ == "__main__":
    import sys
    main(plot_only="--plot-only" in sys.argv)
