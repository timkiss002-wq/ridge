"""
CHƯƠNG TRÌNH MINH HỌA THUẬT TOÁN HỒI QUY RIDGE
Chạy:  streamlit run app.py     (hoặc nháy đúp run_app.bat trên Windows)

Giao diện gồm thanh bên (nhập dữ liệu + tham số) và 7 tab:
  1. Ví dụ tính tay   2. Dữ liệu   3. Các bước tính toán   4. Kết quả & trực quan
  5. Khảo sát α       6. So sánh mô hình   7. Hướng dẫn
"""
import io

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import app_core as core

st.set_page_config(page_title="Hồi quy Ridge", page_icon="📈", layout="wide")

BLUE, ORANGE, GREEN, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "axes.edgecolor": "#b5b4ae", "legend.frameon": False})


def fmt(x, d=4):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def matrix_latex(M, d=4):
    rows = [" & ".join(fmt(v, d).replace(",", "{,}") for v in r) for r in np.atleast_2d(M)]
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


# ======================================================================
# THANH BÊN: dữ liệu + tham số
# ======================================================================
st.sidebar.title("📈 Hồi quy Ridge")
st.sidebar.caption("Nhập dữ liệu và tham số, rồi bấm **Chạy thuật toán**.")

st.sidebar.header("1. Dữ liệu")
source = st.sidebar.radio("Nguồn dữ liệu",
                          ["Diabetes (scikit-learn)", "Mô phỏng đa cộng tuyến", "Tải file CSV"])
beta_true = None
if source == "Diabetes (scikit-learn)":
    X, y = core.load_diabetes_df()
elif source == "Mô phỏng đa cộng tuyến":
    c1, c2 = st.sidebar.columns(2)
    n_sim = c1.number_input("Số mẫu n", 20, 5000, 100, step=10)
    p_sim = c2.number_input("Số biến p", 2, 50, 10)
    rho = st.sidebar.slider("Tương quan giữa các biến ρ", 0.0, 0.99, 0.9, 0.01)
    c3, c4 = st.sidebar.columns(2)
    sigma = c3.number_input("Độ lệch chuẩn nhiễu σ", 0.1, 50.0, 3.0)
    seed_sim = c4.number_input("Seed sinh dữ liệu", 0, 10_000, 42)
    X, y, beta_true = core.simulate(int(n_sim), int(p_sim), rho, sigma, int(seed_sim))
else:
    up = st.sidebar.file_uploader("Chọn file CSV (dòng đầu là tên cột)", type=["csv"])
    if up is None:
        st.sidebar.info("Chưa có file → tạm dùng Diabetes.")
        X, y = core.load_diabetes_df()
    else:
        df = pd.read_csv(up)
        num = df.select_dtypes("number").columns.tolist()
        target = st.sidebar.selectbox("Cột mục tiêu y", num, index=len(num) - 1)
        feats = st.sidebar.multiselect("Các cột đặc trưng X", [c for c in num if c != target],
                                       default=[c for c in num if c != target])
        d = df[feats + [target]].dropna()
        if len(d) < len(df):
            st.sidebar.warning(f"Đã bỏ {len(df) - len(d)} dòng có giá trị thiếu.")
        X, y = d[feats], d[target]

st.sidebar.header("2. Tham số thuật toán")
with st.sidebar.form("params"):
    log_alpha = st.slider("log₁₀(α)", -4.0, 4.0, 0.0, 0.1,
                          help="α = 10^giá trị này. Kéo sang phải để phạt mạnh hơn.")
    alpha_in = st.number_input("hoặc nhập trực tiếp α (để 0 = dùng thanh trượt)", 0.0, 1e6, 0.0,
                               format="%.6f")
    solver = st.selectbox("Phương pháp giải", ["cholesky", "svd", "dual", "gd"],
                          format_func=lambda s: {"cholesky": "Nghiệm đóng – Cholesky",
                                                 "svd": "Nghiệm đóng – SVD",
                                                 "dual": "Dạng đối ngẫu (p ≫ n)",
                                                 "gd": "Gradient descent (lặp)"}[s])
    standardize = st.checkbox("Chuẩn hóa đặc trưng (khuyến nghị)", True)
    fit_intercept = st.checkbox("Có hệ số chặn β₀ (không bị phạt)", True)
    test_size = st.slider("Tỉ lệ tập kiểm tra", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed chia dữ liệu", 0, 10_000, 42)
    with st.expander("Tùy chọn gradient descent"):
        auto_lr = st.checkbox("Tự chọn bước học 1/L", True)
        lr = st.number_input("Bước học η", 1e-8, 1.0, 1e-3, format="%.6f")
        max_iter = st.number_input("Số vòng lặp tối đa", 100, 500_000, 20_000, step=1000)
    run = st.form_submit_button("▶ Chạy thuật toán", type="primary", use_container_width=True)

alpha = alpha_in if alpha_in > 0 else float(10 ** log_alpha)
params = dict(alpha=alpha, solver=solver, standardize=standardize, fit_intercept=fit_intercept,
              test_size=test_size, seed=int(seed), lr=None if auto_lr else lr, max_iter=int(max_iter))
key = (source, X.shape, float(np.asarray(X, float).sum()), tuple(sorted(params.items())))
if run or "S" not in st.session_state:
    with st.spinner("Đang tính..."):
        st.session_state.S = core.run_ridge_steps(X, y, **params)
        st.session_state.key = key
S = st.session_state.S
if st.session_state.get("key") != key:
    st.sidebar.warning("Tham số đã đổi — bấm **Chạy thuật toán** để cập nhật.")
st.sidebar.success(f"α đang dùng = {alpha:.6g}")

# ======================================================================
st.title("Chương trình minh họa thuật toán hồi quy Ridge")
st.caption("Bài tập lớn — cài đặt Ridge từ đầu bằng NumPy (lớp RidgeScratch), "
           "đối chiếu với OLS, Lasso, Elastic Net.")
tabs = st.tabs(["① Ví dụ tính tay", "② Dữ liệu", "③ Các bước tính toán", "④ Kết quả & trực quan",
                "⑤ Khảo sát α", "⑥ So sánh mô hình", "ℹ️ Hướng dẫn"])

# ----------------------------------------------------------------------
# TAB 1 — Ví dụ tính tay
# ----------------------------------------------------------------------
with tabs[0]:
    st.subheader("Ví dụ tính tay với 2 biến gần trùng nhau")
    st.write("Nhập ma trận **XᵀX** (đối xứng) và vectơ **Xᵀy** của dữ liệu đã chuẩn hóa, chọn α. "
             "Chương trình tính từng bước nghiệm β = (XᵀX + αI)⁻¹Xᵀy.")
    c = st.columns(5)
    a11 = c[0].number_input("XᵀX[1,1]", value=1.0, format="%.4f")
    a12 = c[1].number_input("XᵀX[1,2] = XᵀX[2,1]", value=0.99, format="%.4f")
    a22 = c[2].number_input("XᵀX[2,2]", value=1.0, format="%.4f")
    b1 = c[3].number_input("Xᵀy[1]", value=1.0, format="%.4f")
    b2 = c[4].number_input("Xᵀy[2]", value=0.98, format="%.4f")
    ah = st.select_slider("α cho ví dụ", options=[0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], value=0.1)
    XtX2, Xty2 = np.array([[a11, a12], [a12, a22]]), np.array([b1, b2])
    h = core.hand_example(XtX2, Xty2, ah)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Bước 1.** Ma trận đầu vào")
        st.latex(r"X^\top X = " + matrix_latex(XtX2) + r",\quad X^\top y = " + matrix_latex(Xty2[:, None]))
        st.markdown(f"Định thức det(XᵀX) = **{fmt(h['det_XtX'])}**, trị riêng = "
                    f"{', '.join(fmt(v) for v in h['eig_XtX'])}, số điều kiện = **{fmt(h['cond_XtX'], 1)}**")
        st.markdown("**Bước 2.** Cộng α vào đường chéo")
        st.latex(r"A = X^\top X + \alpha I = " + matrix_latex(h["A"]))
        st.markdown(f"det(A) = **{fmt(h['det_A'])}**, trị riêng = {', '.join(fmt(v) for v in h['eig_A'])} "
                    f"(mỗi trị riêng tăng đúng α), số điều kiện = **{fmt(h['cond_A'], 1)}**")
        if h["A_inv"] is not None:
            st.markdown("**Bước 3.** Nghịch đảo")
            st.latex(r"A^{-1} = \frac{1}{\det A}\begin{pmatrix} a_{22} & -a_{12} \\ -a_{12} & a_{11}\end{pmatrix} = "
                     + matrix_latex(h["A_inv"]))
            st.markdown("**Bước 4.** Nghiệm")
            st.latex(r"\hat\beta_{\text{ridge}} = A^{-1}X^\top y = " + matrix_latex(h["beta"][:, None]))
            if h["beta_ols"] is not None:
                st.markdown(f"So sánh OLS (α = 0): β = ({fmt(h['beta_ols'][0])}; {fmt(h['beta_ols'][1])}).")
            if h["shrink"] is not None:
                st.markdown(f"**Bước 5.** Hệ số co theo hướng chính d²/(d²+α) = "
                            f"{', '.join(fmt(v) for v in h['shrink'])} → bậc tự do hiệu dụng df(α) = **{fmt(h['df'])}**")
        else:
            st.error("A suy biến (det = 0): OLS không có nghiệm duy nhất. Hãy tăng α > 0.")
    with right:
        st.markdown("**Bảng: nghiệm theo nhiều α** (đổi Xᵀy[1] ↔ Xᵀy[2] để thấy OLS dao động)")
        tab = core.hand_example_table(XtX2, Xty2, [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10])
        st.dataframe(tab.style.format(precision=4), hide_index=True, use_container_width=True)
        al = np.logspace(-3, 1.5, 120)
        bs = np.array([core.hand_example(XtX2, Xty2, a)["beta"] for a in al])
        fig, ax = plt.subplots(figsize=(4.6, 3))
        ax.plot(al, bs[:, 0], color=BLUE, label="β₁")
        ax.plot(al, bs[:, 1], color=ORANGE, label="β₂")
        ax.axvline(ah, color=INK, ls="--", lw=1)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("α (log)"); ax.set_title("Hệ số theo α", fontsize=10)
        ax.legend()
        show_fig(fig)

# ----------------------------------------------------------------------
# TAB 2 — Dữ liệu
# ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Mô tả dữ liệu")
    m = st.columns(4)
    m[0].metric("Số mẫu n", X.shape[0]); m[1].metric("Số đặc trưng p", X.shape[1])
    m[2].metric("Giá trị thiếu", int(X.isna().sum().sum())); m[3].metric("Biến mục tiêu", y.name)
    st.dataframe(pd.concat([X, y], axis=1).head(10), use_container_width=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        corr = X.corr().to_numpy()
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(X.shape[1]), X.columns, rotation=90); ax.set_yticks(range(X.shape[1]), X.columns)
        ax.grid(False)
        if X.shape[1] <= 12:
            for i in range(X.shape[1]):
                for j in range(X.shape[1]):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(corr[i, j]) > 0.6 else "black")
        fig.colorbar(im, shrink=0.8); ax.set_title("Ma trận tương quan giữa các đặc trưng", fontsize=10)
        show_fig(fig)
    with c2:
        st.markdown("**Hệ số phóng đại phương sai (VIF)** — VIF > 10 thường được xem là đa cộng tuyến mạnh")
        v = core.vif_table(X)
        st.dataframe(v.style.format({"R²_j": "{:.3f}", "VIF": "{:.1f}"})
                     .map(lambda x: "background-color:#fde3d8" if isinstance(x, float) and x > 10 else "",
                          subset=["VIF"]), hide_index=True, use_container_width=True)
        if beta_true is not None:
            st.markdown("**Hệ số thật dùng để sinh dữ liệu mô phỏng**")
            st.dataframe(beta_true.to_frame().T, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 — Các bước tính toán trung gian
# ----------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"Các bước tính toán (α = {alpha:.6g}, phương pháp: {solver})")
    names = S["names"]
    st.markdown(f"**Bước 1 — Chia dữ liệu.** {len(S['idx_train'])} mẫu huấn luyện, "
                f"{len(S['idx_test'])} mẫu kiểm tra (seed = {params['seed']}).")
    st.markdown("**Bước 2 — Chuẩn hóa.** Trung bình và độ lệch chuẩn được tính **chỉ trên tập train**, "
                "rồi áp dụng cho tập test (tránh rò rỉ dữ liệu)." if standardize else
                "**Bước 2 — Chuẩn hóa.** Đã tắt: dùng dữ liệu gốc (hệ số bị phạt không công bằng giữa các biến).")
    with st.expander("Xem tham số chuẩn hóa và 5 dòng dữ liệu sau chuẩn hóa", expanded=False):
        st.dataframe(S["scaler"].style.format(precision=4), hide_index=True, use_container_width=True)
        st.dataframe(S["Z_train_head"].style.format(precision=4), use_container_width=True)
    st.markdown(f"**Bước 3 — Căn giữa.** Trừ trung bình của X và y (ȳ = {fmt(S['ybar'])}) "
                "để intercept không bị phạt.")
    st.markdown("**Bước 4 — Tính XᵀX và Xᵀy.**")
    c1, c2 = st.columns([3, 1])
    c1.dataframe(S["XtX"].style.format(precision=2).background_gradient(cmap="Blues"), use_container_width=True)
    c2.dataframe(S["Xty"].to_frame().style.format(precision=2), use_container_width=True)
    st.markdown(f"**Bước 5 — Cộng αI.** Mọi trị riêng tăng thêm α = {alpha:.6g}. Số điều kiện giảm từ "
                f"**{fmt(S['cond_before'], 1)}** xuống **{fmt(S['cond_after'], 1)}** → hệ phương trình ổn định hơn.")
    st.dataframe(S["eig"].T.style.format(precision=3), use_container_width=True)
    st.markdown("**Bước 6 — Giải hệ (XᵀX + αI)β = Xᵀy.**")
    dtl = S["solver_detail"]
    if solver == "cholesky":
        st.write("Phân rã Cholesky XᵀX + αI = L·Lᵀ (L tam giác dưới), rồi giải hai hệ tam giác Lz = Xᵀy, Lᵀβ = z.")
        st.dataframe(dtl["L"].style.format(precision=3), use_container_width=True)
    elif solver == "svd":
        st.write(f"X = U·D·Vᵀ; β = V·diag(d/(d²+α))·Uᵀy. Hướng có d nhỏ bị co mạnh. "
                 f"Bậc tự do hiệu dụng df(α) = **{fmt(dtl['df'], 3)}** / {S['p']}.")
        cc1, cc2 = st.columns([2, 3])
        cc1.dataframe(dtl["svd"].style.format(precision=4), hide_index=True, use_container_width=True)
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.bar(dtl["svd"]["Hướng chính j"], dtl["svd"]["Hệ số co d²/(d²+α)"], color=BLUE)
        ax.set_ylim(0, 1.05); ax.set_xlabel("Hướng chính j (d giảm dần)"); ax.set_ylabel("Hệ số co")
        ax.set_title("Mỗi hướng chính được giữ lại bao nhiêu phần", fontsize=10)
        with cc2:
            show_fig(fig)
    elif solver == "dual":
        st.write(f"β = Xᵀ(XXᵀ + αI)⁻¹y: chỉ cần giải hệ kích thước {dtl['K_shape'][0]}×{dtl['K_shape'][1]} "
                 f"thay vì {S['p']}×{S['p']} — có lợi khi số biến lớn hơn số mẫu.")
    else:
        st.write(f"Lặp β ← β − η·2(XᵀXβ − Xᵀy + αβ) với η = {dtl['lr']:.3g}. "
                 f"Số vòng lặp: **{dtl['n_iter']}** ({'đã hội tụ' if dtl['converged'] else 'CHƯA hội tụ — tăng số vòng lặp'}).")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(np.arange(len(dtl["loss"])) * 50, dtl["loss"], color=BLUE)
        ax.set_yscale("log"); ax.set_xlabel("Vòng lặp"); ax.set_ylabel("J(β)")
        ax.set_title("Hàm mục tiêu giảm dần qua các vòng lặp", fontsize=10)
        show_fig(fig)
    st.markdown(f"**Bước 7 — Hệ số và intercept.** β₀ = ȳ − x̄ᵀβ = **{fmt(S['intercept_scaled'])}**")
    coef_tab = pd.DataFrame({"β Ridge (đã chuẩn hóa)": S["coef_scaled"], "β OLS (đã chuẩn hóa)": S["coef_ols_scaled"],
                             "β Ridge (thang đo gốc)": S["coef_original"]})
    st.dataframe(coef_tab.style.format(precision=4), use_container_width=True)
    st.caption(f"Intercept ở thang đo gốc = {fmt(S['intercept_original'])}. "
               f"‖β‖ Ridge = {fmt(S['coef_norm']['Ridge'], 3)}, ‖β‖ OLS = {fmt(S['coef_norm']['OLS'], 3)}."
               + ("" if S["ols_full_rank"] else " ⚠️ Ma trận thiết kế không đủ hạng: OLS không có nghiệm duy nhất."))
    st.markdown("**Bước 8 — Dự đoán trên tập kiểm tra** (10 dòng đầu)")
    st.dataframe(S["pred_test"].head(10).style.format(precision=2), use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4 — Kết quả & trực quan
# ----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Kết quả cuối cùng")
    mt = S["metrics"].set_index(["Mô hình", "Tập"])
    r_te, o_te = mt.loc[("Ridge", "Test")], mt.loc[("OLS", "Test")]
    k = st.columns(4)
    k[0].metric("RMSE test – Ridge", fmt(r_te["RMSE"], 3), fmt(r_te["RMSE"] - o_te["RMSE"], 3) + " so với OLS",
                delta_color="inverse")
    k[1].metric("MAE test – Ridge", fmt(r_te["MAE"], 3), fmt(r_te["MAE"] - o_te["MAE"], 3) + " so với OLS",
                delta_color="inverse")
    k[2].metric("R² test – Ridge", fmt(r_te["R²"], 4), fmt(r_te["R²"] - o_te["R²"], 4) + " so với OLS")
    k[3].metric("α", f"{alpha:.4g}")
    st.dataframe(S["metrics"].style.format(precision=4), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        idx = np.arange(S["p"])
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.barh(idx + 0.2, S["coef_ols_scaled"], height=0.38, color=GRAY, label="OLS")
        ax.barh(idx - 0.2, S["coef_scaled"], height=0.38, color=BLUE, label="Ridge")
        ax.set_yticks(idx, S["names"]); ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("Hệ số (dữ liệu đã chuẩn hóa)"); ax.legend()
        ax.set_title("Hệ số Ridge so với OLS", fontsize=10)
        if beta_true is not None and standardize is False:
            ax.scatter(beta_true.values, idx, marker="|", s=120, color=ORANGE, label="β thật")
        show_fig(fig)
    with c2:
        pt = S["pred_test"]
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.scatter(pt["y thật"], pt["ŷ Ridge"], s=16, color=BLUE, alpha=0.8, label="Ridge")
        lo, hi = pt[["y thật", "ŷ Ridge"]].min().min(), pt[["y thật", "ŷ Ridge"]].max().max()
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="Dự đoán hoàn hảo")
        ax.set_xlabel("y thật (tập test)"); ax.set_ylabel("y dự đoán"); ax.legend()
        ax.set_title("Giá trị dự đoán so với giá trị thật", fontsize=10)
        show_fig(fig)
    out = io.StringIO()
    pd.DataFrame({"feature": S["names"], "coef_scaled": S["coef_scaled"].values,
                  "coef_original": S["coef_original"].values}).to_csv(out, index=False)
    st.download_button("⬇ Tải hệ số (CSV)", out.getvalue(), "ridge_coefficients.csv", "text/csv")

# ----------------------------------------------------------------------
# TAB 5 — Khảo sát α
# ----------------------------------------------------------------------
with tabs[4]:
    st.subheader("Khảo sát α bằng K-fold cross-validation (chỉ trên tập train)")
    c = st.columns(4)
    lo = c[0].number_input("log₁₀ α nhỏ nhất", -6.0, 0.0, -4.0)
    hi = c[1].number_input("log₁₀ α lớn nhất", 0.0, 8.0, 4.0)
    npts = c[2].number_input("Số giá trị α", 10, 200, 50)
    kf = c[3].number_input("Số fold K", 2, 20, 5)
    if st.button("▶ Chạy khảo sát α"):
        alphas = np.logspace(lo, hi, int(npts))
        st.session_state.survey = core.alpha_survey(X, y, alphas, int(kf), int(seed), standardize, test_size)
    if "survey" in st.session_state:
        res, path, a_best, a_1se = st.session_state.survey
        k = st.columns(3)
        k[0].metric("α tốt nhất (RMSE CV nhỏ nhất)", f"{a_best:.4g}")
        k[1].metric("α theo quy tắc 1 sai số chuẩn", f"{a_1se:.4g}")
        k[2].metric("RMSE CV nhỏ nhất", fmt(res["RMSE_CV"].min(), 3))
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            ax.fill_between(res["alpha"], res["RMSE_CV"] - res["RMSE_CV_std"], res["RMSE_CV"] + res["RMSE_CV_std"],
                            color=BLUE, alpha=0.12, lw=0)
            ax.plot(res["alpha"], res["RMSE_CV"], color=BLUE, label="RMSE cross-validation")
            ax.plot(res["alpha"], res["RMSE_train"], color=ORANGE, label="RMSE huấn luyện")
            ax.axvline(a_best, color=INK, ls="--", lw=1)
            ax.set_xscale("log"); ax.set_xlabel("α (log)"); ax.set_ylabel("RMSE"); ax.legend()
            ax.set_title("Sai số theo α (dải mờ: ±1 độ lệch chuẩn giữa các fold)", fontsize=10)
            show_fig(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            cols = path.columns[1:]
            pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
            for j, col in enumerate(cols):
                ax.plot(path["alpha"], path[col], color=pal[j] if j < 8 else GRAY, lw=1.6 if j < 8 else 1)
                ax.annotate(col, (path["alpha"].iloc[0], path[col].iloc[0]), xytext=(-4, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=7, color=INK)
            ax.axvline(a_best, color=INK, ls="--", lw=1); ax.axhline(0, color=GRAY, lw=0.8)
            ax.set_xscale("log"); ax.set_xlabel("α (log)"); ax.set_ylabel("Hệ số")
            ax.set_xlim(path["alpha"].iloc[0] / 30, path["alpha"].iloc[-1] * 1.5)
            ax.set_title("Ridge trace: hệ số co dần về 0 khi α tăng", fontsize=10)
            show_fig(fig)
        st.dataframe(res.style.format(precision=4), hide_index=True, use_container_width=True, height=240)
        st.info("Đặt α tốt nhất vào ô 'nhập trực tiếp α' ở thanh bên rồi bấm Chạy để xem chi tiết mô hình đó.")

# ----------------------------------------------------------------------
# TAB 6 — So sánh mô hình
# ----------------------------------------------------------------------
with tabs[5]:
    st.subheader("So sánh OLS, Ridge, Lasso, Elastic Net (scikit-learn)")
    st.write("Mỗi mô hình chọn siêu tham số bằng 5-fold CV trên tập train, rồi đánh giá đúng một lần trên tập test.")
    if st.button("▶ Chạy so sánh (mất vài giây)"):
        with st.spinner("Đang huấn luyện 4 mô hình..."):
            st.session_state.cmp = core.compare_models(X, y, test_size, int(seed))
    if "cmp" in st.session_state:
        tab, coefs = st.session_state.cmp
        st.dataframe(tab.style.format(precision=4).highlight_min(subset=["RMSE test"], color="#d6ecdf"),
                     hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            colors = [GRAY, BLUE, ORANGE, GREEN]
            for i, (v, col) in enumerate(zip(tab["RMSE test"], colors)):
                ax.plot([v], [i], "o", ms=9, color=col)
                ax.annotate(fmt(v, 3), (v, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
            ax.set_yticks(range(len(tab)), tab["Mô hình"]); ax.invert_yaxis()
            span = tab["RMSE test"].max() - tab["RMSE test"].min()
            ax.set_xlim(tab["RMSE test"].min() - span - 0.05, tab["RMSE test"].max() + span + 0.15)
            ax.set_xlabel("RMSE test (càng nhỏ càng tốt)"); ax.set_title("Sai số trên tập test", fontsize=10)
            show_fig(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            idx = np.arange(len(coefs))
            for k_, (mname, col) in enumerate(zip(coefs.columns, colors)):
                ax.barh(idx + (k_ - 1.5) * 0.2, coefs[mname], height=0.2, color=col, label=mname)
            ax.set_yticks(idx, coefs.index); ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8)
            ax.legend(fontsize=7); ax.set_title("Hệ số của 4 mô hình (đã chuẩn hóa)", fontsize=10)
            show_fig(fig)

# ----------------------------------------------------------------------
# TAB 7 — Hướng dẫn
# ----------------------------------------------------------------------
with tabs[6]:
    st.subheader("Hướng dẫn nhanh")
    st.markdown("""
1. **Chọn dữ liệu** ở thanh bên: Diabetes (có sẵn), dữ liệu mô phỏng (chỉnh n, p, ρ, σ) hoặc file CSV của bạn.
2. **Nhập tham số**: α (thanh trượt log hoặc nhập số), phương pháp giải, chuẩn hóa, intercept, tỉ lệ test, seed.
3. Bấm **▶ Chạy thuật toán**.
4. Xem **③ Các bước tính toán** (kết quả trung gian) và **④ Kết quả & trực quan** (kết quả cuối, biểu đồ).
5. Dùng **⑤ Khảo sát α** để tìm α tốt nhất bằng cross-validation; **⑥ So sánh mô hình** để đối chiếu với OLS, Lasso, Elastic Net.
6. Tab **① Ví dụ tính tay** tính lại ví dụ ma trận 2×2 trong báo cáo, cho phép nhập số tùy ý.

**Công thức:** β̂ = (XᵀX + αI)⁻¹Xᵀy; α = 0 trở về OLS; α càng lớn hệ số càng co về 0.
""")
