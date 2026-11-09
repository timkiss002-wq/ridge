"""
TN2 — Ảnh hưởng của alpha đến Ridge (Bảng 5.4, Hình 5.3).
Với mỗi alpha: RMSE train, RMSE CV (5x5), df(alpha), chuẩn ||beta||.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import common as U
from . import config as C
from .ridge_scratch import effective_df


def main():
    print("[TN2] Quét alpha cho Ridge")
    X, y = U.load_data()
    X_tr, _, y_tr, _ = U.split(X, y)            # tập test KHÔNG được dùng ở đây
    cv = RepeatedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=C.SEED)
    Xs_tr = StandardScaler().fit_transform(X_tr)
    df_alpha = effective_df(Xs_tr, C.RIDGE_ALPHAS)

    rows = []
    for a, dfa in zip(C.RIDGE_ALPHAS, df_alpha):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=a))
        s = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error",
                           return_train_score=True, n_jobs=-1)
        coef = pipe.fit(X_tr, y_tr)[-1].coef_
        rows.append({"alpha": a, "df": dfa,
                     "RMSE_train_mean": -s["train_score"].mean(),
                     "RMSE_CV_mean": -s["test_score"].mean(),
                     "RMSE_CV_std": s["test_score"].std(),
                     "RMSE_CV_se": s["test_score"].std() / np.sqrt(len(s["test_score"])),
                     "coef_norm": np.linalg.norm(coef)})
    res = pd.DataFrame(rows)

    i_best = res["RMSE_CV_mean"].idxmin()
    thr = res.loc[i_best, "RMSE_CV_mean"] + res.loc[i_best, "RMSE_CV_se"]
    i_1se = res.index[res["RMSE_CV_mean"] <= thr].max()      # alpha lớn nhất trong 1 SE
    res["is_best"] = res.index == i_best
    res["is_1se"] = res.index == i_1se
    U.save_csv(res, "tn2_alpha_curve.csv")

    a_best, a_1se = res.loc[i_best, "alpha"], res.loc[i_1se, "alpha"]
    print(f"  alpha tốt nhất (CV) = {a_best:.4g}, RMSE CV = {res.loc[i_best, 'RMSE_CV_mean']:.3f}"
          f" | alpha theo quy tắc 1-SE = {a_1se:.4g}")

    # Bảng 5.4: các alpha đại diện
    rep = [1e-4, 1e-2, 1, 1e2, 1e4]
    pick = res.iloc[[int(np.argmin(np.abs(np.log10(res["alpha"]) - np.log10(r)))) for r in rep]]
    U.save_csv(pick.drop(columns=["is_best", "is_1se"]), "tn2_table_5_4.csv")

    # Hình 5.3
    U.style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(res["alpha"], res["RMSE_CV_mean"] - res["RMSE_CV_std"],
                    res["RMSE_CV_mean"] + res["RMSE_CV_std"], color=C.COLORS["Ridge"],
                    alpha=0.12, linewidth=0, label="CV ± 1 độ lệch chuẩn giữa các fold")
    ax.plot(res["alpha"], res["RMSE_CV_mean"], color=C.COLORS["Ridge"], label="RMSE cross-validation")
    ax.plot(res["alpha"], res["RMSE_train_mean"], color=C.COLORS["Lasso"], label="RMSE huấn luyện")
    ax.axvline(a_best, color=C.TEXT2, linestyle="--", linewidth=1)
    ax.annotate(f"α tốt nhất ≈ {a_best:.3g}".replace(".", ","), (a_best, 63), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=C.TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("α (thang log)")
    ax.set_ylabel("RMSE (đơn vị biến mục tiêu)")
    ax.set_title("Sai số huấn luyện và cross-validation của Ridge theo α")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    U.save(fig, "hinh_5_3_alpha_curve.png")


if __name__ == "__main__":
    main()
