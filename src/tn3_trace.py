"""
TN3 — Ridge trace và đường hệ số Lasso trên Diabetes (Hình 5.4).
Hệ số tính trên dữ liệu train đã chuẩn hóa (nên so sánh được độ lớn giữa các biến).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from . import common as U
from . import config as C
from .ridge_scratch import ridge_path_svd

# Các biến được tô màu (chỉ số huyết thanh + bmi); các biến còn lại màu xám.
HIGHLIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
             "s4": "#eda100", "s5": "#e87ba4", "bmi": "#6250d6"}


def main():
    print("[TN3] Ridge trace và đường hệ số Lasso")
    X, y = U.load_data()
    X_tr, _, y_tr, _ = U.split(X, y)
    Xs = StandardScaler().fit_transform(X_tr)
    names = list(X.columns)

    ridge_coefs = ridge_path_svd(Xs, y_tr.to_numpy(), C.RIDGE_ALPHAS)        # 1 lần SVD
    lasso_coefs = np.array([Lasso(alpha=a, max_iter=100_000).fit(Xs, y_tr).coef_
                            for a in C.LASSO_ALPHAS])

    r = pd.DataFrame(ridge_coefs, columns=names); r.insert(0, "alpha", C.RIDGE_ALPHAS)
    l = pd.DataFrame(lasso_coefs, columns=names); l.insert(0, "alpha", C.LASSO_ALPHAS)
    U.save_csv(r, "tn3_ridge_trace.csv")
    U.save_csv(l, "tn3_lasso_path.csv")

    # Thứ tự biến bị Lasso đưa về 0 khi alpha tăng
    first_zero = {}
    for j, n in enumerate(names):
        nz = np.abs(lasso_coefs[:, j]) > 1e-10
        first_zero[n] = C.LASSO_ALPHAS[np.argmax(~nz)] if (~nz).any() else np.nan
    print("  Lasso: alpha đầu tiên mà hệ số = 0:",
          ", ".join(f"{k}={v:.3g}" for k, v in sorted(first_zero.items(), key=lambda t: t[1])))
    print("  Ridge ở alpha nhỏ nhất:", dict(zip(names, ridge_coefs[0].round(1))))
    print("  Ridge ở alpha lớn nhất:", dict(zip(names, ridge_coefs[-1].round(2))))

    # Hình 5.4
    U.style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, coefs, alphas, title in [
        (axes[0], ridge_coefs, C.RIDGE_ALPHAS, "Ridge: hệ số co dần, không về đúng 0"),
        (axes[1], lasso_coefs, C.LASSO_ALPHAS, "Lasso: hệ số lần lượt về đúng 0"),
    ]:
        for j, n in enumerate(names):
            color = HIGHLIGHT.get(n, C.GRAY)
            lw = 2 if n in HIGHLIGHT else 1.2
            ax.plot(alphas, coefs[:, j], color=color, linewidth=lw, zorder=3 if n in HIGHLIGHT else 2)
        ax.axhline(0, color=C.TEXT2, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("α (thang log)")
        ax.set_title(title, fontsize=10)
        # Nhãn trực tiếp ở mép trái, giãn cách để không chồng nhau
        y0 = coefs[0]
        order = np.argsort(y0)
        placed, gap = [], (y0.max() - y0.min()) * 0.045
        for j in order:
            yy = y0[j]
            if placed and yy - placed[-1] < gap:
                yy = placed[-1] + gap
            placed.append(yy)
            ax.annotate(names[j], (alphas[0], y0[j]), xytext=(alphas[0] / 1.6, yy),
                        textcoords="data", ha="right", va="center", fontsize=8.5,
                        color=C.TEXT if names[j] in HIGHLIGHT else C.TEXT2)
        ax.set_xlim(alphas[0] / 12, alphas[-1] * 1.5)
    axes[0].set_ylabel("Hệ số (trên dữ liệu đã chuẩn hóa)")
    fig.suptitle("Hệ số hồi quy theo α trên Diabetes (màu: bmi và chỉ số huyết thanh s1–s5)",
                 fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.04)
    U.save(fig, "hinh_5_4_ridge_trace.png")


if __name__ == "__main__":
    main()
