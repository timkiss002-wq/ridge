"""Hàm dùng chung: tải dữ liệu, chỉ số đánh giá, mô hình, định dạng biểu đồ."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def load_data():
    """Diabetes bản GỐC (scaled=False) -> chuẩn hóa sẽ làm trong pipeline (tránh rò rỉ)."""
    ds = load_diabetes(scaled=False, as_frame=True)
    return ds.data, ds.target


def split(X, y, seed=C.SEED):
    return train_test_split(X, y, test_size=C.TEST_SIZE, random_state=seed)


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def model_specs():
    """(estimator, lưới siêu tham số) cho 4 mô hình. Mọi mô hình đều có StandardScaler."""
    return {
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"ridge__alpha": C.RIDGE_ALPHAS}),
        "Lasso": (Lasso(max_iter=100_000), {"lasso__alpha": C.LASSO_ALPHAS}),
        "ElasticNet": (ElasticNet(max_iter=100_000),
                       {"elasticnet__alpha": C.EN_ALPHAS,
                        "elasticnet__l1_ratio": C.EN_L1_RATIOS}),
    }


def tune_and_evaluate(name, X_tr, y_tr, X_te, y_te, seed=C.SEED):
    """Chọn siêu tham số bằng CV trên train, huấn luyện lại, đánh giá test đúng 1 lần."""
    est, grid = model_specs()[name]
    pipe = make_pipeline(StandardScaler(), est)
    cv = RepeatedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=seed)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error",
                          n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    i = search.best_index_
    row = {"model": name}
    for k, v in search.best_params_.items():
        row[k.split("__")[1]] = v
    row["RMSE_CV_mean"] = -search.cv_results_["mean_test_score"][i]
    row["RMSE_CV_std"] = search.cv_results_["std_test_score"][i]
    for prefix, (Xs, ys) in {"train": (X_tr, y_tr), "test": (X_te, y_te)}.items():
        for k, v in metrics(ys, best.predict(Xs)).items():
            row[f"{k}_{prefix}"] = v
    coef = best[-1].coef_
    row["n_nonzero_coef"] = int(np.sum(np.abs(coef) > 1e-10))
    return row, best


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",      # hỗ trợ tiếng Việt
        "font.size": 10,
        "axes.edgecolor": C.GRAY,
        "axes.labelcolor": C.TEXT2,
        "axes.titlecolor": C.TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": C.TEXT2,
        "ytick.color": C.TEXT2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def save(fig, name):
    path = C.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.relative_to(C.ROOT)}")


def save_csv(df: pd.DataFrame, name):
    path = C.RESULTS / name
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"  -> {path.relative_to(C.ROOT)}")
