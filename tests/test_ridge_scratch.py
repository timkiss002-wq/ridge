"""Kiểm thử RidgeScratch. Chạy: python -m tests.test_ridge_scratch  (hoặc pytest nếu có)."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from src.ridge_scratch import RidgeScratch, effective_df, ridge_path_svd


def _data(n=60, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.01 * rng.normal(size=n)          # 2 cột gần trùng nhau
    y = X @ rng.normal(size=p) + 5 + rng.normal(size=n)
    return X, y


def test_all_solvers_match_sklearn():
    X, y = _data()
    for alpha in [0.01, 1.0, 50.0]:
        ref = Ridge(alpha=alpha).fit(X, y)
        for s in RidgeScratch.SOLVERS:
            m = RidgeScratch(alpha=alpha, solver=s).fit(X, y)
            assert np.allclose(m.coef_, ref.coef_, atol=1e-5), (s, alpha)
            assert np.isclose(m.intercept_, ref.intercept_, atol=1e-5), (s, alpha)


def test_alpha_zero_equals_ols():
    X, y = _data()
    m = RidgeScratch(alpha=0.0, solver="svd").fit(X, y)
    assert np.allclose(m.coef_, LinearRegression().fit(X, y).coef_, atol=1e-6)


def test_p_greater_than_n_solvable():
    X, y = _data(n=20, p=50)
    a = RidgeScratch(alpha=1.0, solver="dual").fit(X, y).coef_
    b = RidgeScratch(alpha=1.0, solver="cholesky").fit(X, y).coef_
    assert np.allclose(a, b, atol=1e-8)


def test_shrinkage_monotone_and_df():
    X, y = _data()
    alphas = np.logspace(-3, 4, 30)
    norms = np.linalg.norm(ridge_path_svd(X, y, alphas), axis=1)
    assert np.all(np.diff(norms) <= 1e-12)                  # ||β|| giảm khi α tăng
    df = effective_df(X, [0.0, 1e12])
    assert np.isclose(df[0], X.shape[1]) and df[1] < 1e-6   # df(0) = p, df(∞) = 0


def test_hand_example_table_3_1():
    XtX = np.array([[1, 0.99], [0.99, 1]])
    b = np.linalg.solve(XtX + 0.1 * np.eye(2), [1, 0.98])
    assert np.allclose(b, [0.5646, 0.3828], atol=1e-4)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
