"""
Cài đặt hồi quy Ridge từ đầu bằng NumPy (không dùng scikit-learn).

Hàm mục tiêu (cùng dạng với sklearn.linear_model.Ridge, xem Mục 3.1 của báo cáo):

    J(beta) = ||y - X beta||^2 + alpha * ||beta||^2       (intercept KHÔNG bị phạt)

Bốn cách giải (Mục 3.2 và 3.8):
    "cholesky" : giải (XᵀX + αI) β = Xᵀy bằng phân rã Cholesky      -> tốt khi p vừa phải
    "svd"      : β = V diag(d / (d² + α)) Uᵀ y                       -> ổn định số học, cho cả đường nghiệm
    "dual"     : β = Xᵀ (XXᵀ + αI)⁻¹ y                               -> tốt khi p >> n
    "gd"       : gradient descent trên J(β)                           -> minh họa phương pháp lặp
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class RidgeScratch:
    """Hồi quy Ridge tự cài đặt.

    Tham số
    -------
    alpha : float >= 0
        Hệ số phạt L2. alpha = 0 trở về OLS (chỉ giải được khi XᵀX khả nghịch).
    solver : {"cholesky", "svd", "dual", "gd"}
    fit_intercept : bool
        Nếu True: căn giữa X và y, giải Ridge không intercept, rồi tính
        intercept = mean(y) - mean(X) @ coef  (Mục 3.7) -> intercept không bị phạt.
    lr, max_iter, tol : chỉ dùng cho solver="gd".
    """

    SOLVERS = ("cholesky", "svd", "dual", "gd")

    def __init__(self, alpha=1.0, solver="cholesky", fit_intercept=True,
                 lr=None, max_iter=100_000, tol=1e-10):
        if alpha < 0:
            raise ValueError("alpha phải >= 0")
        if solver not in self.SOLVERS:
            raise ValueError(f"solver phải thuộc {self.SOLVERS}")
        self.alpha = float(alpha)
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------ #
    # Huấn luyện
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.fit_intercept:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean()
        else:
            self.x_mean_ = np.zeros(X.shape[1])
            self.y_mean_ = 0.0
        Xc = X - self.x_mean_
        yc = y - self.y_mean_

        solve = getattr(self, f"_solve_{self.solver}")
        self.coef_ = solve(Xc, yc)
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def _solve_cholesky(self, X, y):
        # (XᵀX + αI) là ma trận đối xứng xác định dương khi α > 0 (Mục 3.2, Bước 4)
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        b = X.T @ y
        return cho_solve(cho_factor(A), b)

    def _solve_svd(self, X, y):
        # X = U D Vᵀ  =>  β = V diag(d/(d²+α)) Uᵀ y   (Mục 3.5)
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        d_safe = np.where(d > 1e-12 * d.max(), d, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(d_safe > 0, d_safe / (d_safe ** 2 + self.alpha), 0.0)
        return Vt.T @ (factor * (U.T @ y))

    def _solve_dual(self, X, y):
        # β = Xᵀ (XXᵀ + αI)⁻¹ y : chỉ phải giải hệ n×n (Mục 3.8)
        K = X @ X.T + self.alpha * np.eye(X.shape[0])
        return X.T @ cho_solve(cho_factor(K), y)

    def _solve_gd(self, X, y):
        # ∇J = 2(XᵀXβ − Xᵀy + αβ). Bước học 1/L, L = 2(σ_max² + α) là hằng số Lipschitz
        XtX = X.T @ X
        Xty = X.T @ y
        L = 2 * (np.linalg.eigvalsh(XtX).max() + self.alpha)
        lr = self.lr if self.lr is not None else 1.0 / L
        beta = np.zeros(X.shape[1])
        self.loss_history_ = []
        for it in range(self.max_iter):
            grad = 2 * (XtX @ beta - Xty + self.alpha * beta)
            beta_new = beta - lr * grad
            if it % 50 == 0:
                r = y - X @ beta_new
                self.loss_history_.append(r @ r + self.alpha * beta_new @ beta_new)
            if np.linalg.norm(beta_new - beta) <= self.tol * (1 + np.linalg.norm(beta)):
                beta = beta_new
                break
            beta = beta_new
        self.n_iter_ = it + 1
        return beta

    # ------------------------------------------------------------------ #
    # Dự đoán và đánh giá
    # ------------------------------------------------------------------ #
    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R² trên (X, y)."""
        y = np.asarray(y, dtype=float).ravel()
        resid = y - self.predict(X)
        return 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------------- #
# Các hàm tiện ích dựa trên SVD (Mục 3.5)
# ---------------------------------------------------------------------- #
def ridge_path_svd(X, y, alphas, fit_intercept=True):
    """Toàn bộ đường hệ số Ridge cho nhiều alpha chỉ với MỘT lần SVD.

    Trả về mảng (len(alphas), p).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if fit_intercept:
        X = X - X.mean(axis=0)
        y = y - y.mean()
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    Uty = U.T @ y
    return np.array([Vt.T @ ((d / (d ** 2 + a)) * Uty) for a in alphas])


def effective_df(X, alphas, fit_intercept=True):
    """Bậc tự do hiệu dụng df(α) = Σ d_j² / (d_j² + α)."""
    X = np.asarray(X, dtype=float)
    if fit_intercept:
        X = X - X.mean(axis=0)
    d2 = np.linalg.svd(X, compute_uv=False) ** 2
    return np.array([np.sum(d2 / (d2 + a)) for a in np.atleast_1d(alphas)])


def shrinkage_factors(X, alpha, fit_intercept=True):
    """Hệ số co d_j² / (d_j² + α) của từng hướng chính."""
    X = np.asarray(X, dtype=float)
    if fit_intercept:
        X = X - X.mean(axis=0)
    d2 = np.linalg.svd(X, compute_uv=False) ** 2
    return d2 / (d2 + alpha)


def ridge_from_gram(XtX, Xty, alpha):
    """Giải trực tiếp (XᵀX + αI) β = Xᵀy khi đã có sẵn XᵀX, Xᵀy (dùng cho ví dụ tính tay)."""
    XtX = np.asarray(XtX, dtype=float)
    return np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), np.asarray(Xty, dtype=float))
