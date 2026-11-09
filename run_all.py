"""Chạy toàn bộ thí nghiệm theo thứ tự. Dùng: python run_all.py [--quick]"""
import platform
import sys
import time

import matplotlib
import numpy
import pandas
import scipy
import sklearn

from src import config as C

if "--quick" in sys.argv:          # chạy nhanh để thử: TN4 chỉ 3 seed
    C.N_SEEDS_TN4 = 3

from src import tn0_verify, tn1_compare, tn2_alpha, tn3_trace, tn4_generalization, tn5_simulate

env = (f"Python {platform.python_version()} | numpy {numpy.__version__} | pandas {pandas.__version__} | "
       f"scipy {scipy.__version__} | scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__} | "
       f"{platform.platform()}")
print(env)
(C.RESULTS / "environment.txt").write_text(env + "\n", encoding="utf-8")

t0 = time.perf_counter()
for mod in [tn0_verify, tn1_compare, tn2_alpha, tn3_trace, tn4_generalization, tn5_simulate]:
    t = time.perf_counter()
    mod.main()
    print(f"   ({time.perf_counter() - t:.1f} s)\n")
print(f"Hoàn tất sau {time.perf_counter() - t0:.1f} s. Kết quả: results/, hình: figures/")
