# Bài tập lớn: Thuật toán hồi quy Ridge — mã nguồn

Gồm:

1. **Chương trình giao diện** (`app.py`, Streamlit): nhập dữ liệu và tham số, xem từng bước tính toán
   trung gian, kết quả cuối cùng và biểu đồ.
2. **Cài đặt Ridge từ đầu bằng NumPy** (`src/ridge_scratch.py`) với 4 cách giải:
   Cholesky, SVD, dạng đối ngẫu (p ≫ n) và gradient descent. Đã đối chiếu với `sklearn.linear_model.Ridge`.
3. **Năm thí nghiệm** so sánh OLS, Ridge, Lasso, Elastic Net (dùng scikit-learn), sinh số liệu cho báo cáo.

## Chạy nhanh trên Windows

Yêu cầu: Python 3.10 trở lên đã cài và có trong PATH (khi cài Python, tích "Add python.exe to PATH").

1. Nháy đúp `setup.bat` (chỉ làm một lần: tạo môi trường `.venv` và cài thư viện, cần Internet).
2. Nháy đúp `run_app.bat` → trình duyệt tự mở chương trình tại http://localhost:8501.
3. (Tùy chọn) Nháy đúp `run_experiments.bat` để chạy lại kiểm thử và toàn bộ thí nghiệm (~10 phút).

Đóng cửa sổ dòng lệnh để tắt chương trình.

## Cài đặt và chạy bằng dòng lệnh

```bash
pip install -r requirements.txt
streamlit run app.py                   # mở chương trình giao diện
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m tests.test_ridge_scratch     # kiểm thử cài đặt tự viết (vài giây)
python run_all.py                      # chạy toàn bộ thí nghiệm (~7–10 phút, chủ yếu là TN4)
python run_all.py --quick              # chạy nhanh để thử (TN4 chỉ 3 seed, ~2 phút)
python -m src.tn2_alpha                # hoặc chạy riêng từng thí nghiệm
```

Mọi lệnh chạy từ thư mục gốc `ridge-btl/`. Không cần Internet: dữ liệu Diabetes có sẵn trong scikit-learn.

## Cấu trúc

| Tệp | Nội dung | Đầu ra |
| --- | --- | --- |
| `app.py` | Chương trình giao diện: 7 tab (ví dụ tính tay, dữ liệu, các bước tính, kết quả, khảo sát α, so sánh mô hình, hướng dẫn) | — |
| `src/app_core.py` | Lõi tính toán cho giao diện (trả về kết quả trung gian), không phụ thuộc Streamlit | — |
| `setup.bat`, `run_app.bat`, `run_experiments.bat` | Tệp chạy trên Windows | — |
| `src/ridge_scratch.py` | Lớp `RidgeScratch` (4 solver), `ridge_path_svd`, `effective_df`, `shrinkage_factors` | — |
| `src/config.py` | Seed, tỉ lệ chia, CV 5×5, lưới α, màu | — |
| `src/common.py` | Tải dữ liệu, chỉ số MAE/RMSE/R², pipeline + GridSearchCV | — |
| `src/tn0_verify.py` | Tái lập Bảng 3.1 (ví dụ tính tay); so sánh RidgeScratch với sklearn | `tn0_*.csv` |
| `src/tn1_compare.py` | So sánh 4 mô hình; mô tả dữ liệu, tương quan, VIF (trên train) | `tn1_*.csv` → Bảng 5.1, 5.3 |
| `src/tn2_alpha.py` | Sai số train/CV theo α, df(α), quy tắc 1-SE | `tn2_*.csv`, Hình 5.3 → Bảng 5.4 |
| `src/tn3_trace.py` | Ridge trace và đường hệ số Lasso | `tn3_*.csv`, Hình 5.4 |
| `src/tn4_generalization.py` | Lặp lại TN1 với 20 seed | `tn4_*.csv`, Hình 5.5 |
| `src/tn5_simulate.py` | **Dữ liệu mô phỏng**: (a) đa thức bậc 9, (b) đa cộng tuyến ρ ∈ {0; 0,5; 0,9; 0,99} | `tn5*.csv`, Hình 5.2, 5.6 |
| `tests/test_ridge_scratch.py` | 5 kiểm thử (khớp sklearn, α=0 ≡ OLS, p > n, co đơn điệu, Bảng 3.1) | — |
| `run_all.py` | Chạy tất cả, ghi `results/environment.txt` | — |

## Nguyên tắc đảm bảo đánh giá công bằng

- `StandardScaler` luôn nằm trong `Pipeline` → tham số chuẩn hóa chỉ học trên fold huấn luyện.
- Diabetes được tải bằng `load_diabetes(scaled=False)` (bản mặc định đã chuẩn hóa trên toàn bộ dữ liệu).
- α (và `l1_ratio`) chọn bằng `RepeatedKFold(5, 5)` **chỉ trên tập train**; tập test (20%) được dùng đúng một lần.
- Seed cố định (`SEED = 42`; TN4 dùng seed 0–19; TN5 dùng `numpy.random.default_rng(42)`).
- Thang α của `Ridge` khác `Lasso`/`ElasticNet` (sklearn chia sai số cho 2n ở hai lớp sau), nên không so sánh trực tiếp giá trị α giữa các mô hình.
- Ở TN5a, ba giá trị α (0; 0,01; 1) được **chọn thủ công để minh họa**, không tinh chỉnh.

## Dùng thử lớp RidgeScratch

```python
import numpy as np
from src.ridge_scratch import RidgeScratch

X = np.random.default_rng(0).normal(size=(100, 5))
y = X @ np.array([1, 2, 0, 0, -1]) + 3 + np.random.default_rng(1).normal(size=100)

m = RidgeScratch(alpha=1.0, solver="svd").fit(X, y)
print(m.coef_, m.intercept_, m.score(X, y))
```
