"""Cấu hình chung cho mọi thí nghiệm (Bảng 5.2 của báo cáo)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 42                      # seed chính cho chia train/test và CV
TEST_SIZE = 0.2                # 80% train / 20% test
CV_SPLITS, CV_REPEATS = 5, 5   # RepeatedKFold 5x5, chỉ trên tập train
N_SEEDS_TN4 = 20               # TN4: lặp lại với seed 0..19

# Lưới alpha. Lưu ý (Mục 3.1): sklearn.Ridge dùng ||y-Xb||² + α||b||²,
# còn Lasso/ElasticNet dùng (1/2n)||y-Xb||² + α·penalty -> thang alpha khác nhau.
RIDGE_ALPHAS = np.logspace(-4, 4, 50)
LASSO_ALPHAS = np.logspace(-3, 2, 50)
EN_ALPHAS = np.logspace(-3, 2, 25)
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Bảng màu (thứ tự cố định, không lặp vòng)
COLORS = {
    "OLS": "#52514e",
    "Ridge": "#2a78d6",
    "Lasso": "#eb6834",
    "ElasticNet": "#1baf7a",
}
GRAY = "#b5b4ae"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
