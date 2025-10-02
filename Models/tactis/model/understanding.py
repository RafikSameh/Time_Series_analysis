import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
import pandas as pd

# ----------------------------
# 1. Create 3 fake time series
# ----------------------------
T = 30
time = pd.date_range("2020-01-01", periods=T, freq="D")

def make_series(offset):
    target = 50 + 10*np.sin(np.arange(T)/3 + offset) + np.random.normal(0, 1, T)
    covar1 = 20 + 5*np.cos(np.arange(T)/10 + offset)
    covar2 = (np.arange(T) % 24)
    return {
        "start": time[0],
        "target": target.astype(np.float32),
        "feat_dynamic_real": np.stack([covar1, covar2], axis=0).astype(np.float32)
    }

series_list = [make_series(0), make_series(1), make_series(2)]

# ----------------------------
# 2. Plot before grouping
# ----------------------------
plt.figure(figsize=(10, 5))
for i, s in enumerate(series_list):
    plt.plot(time, s["target"], label=f"Series {i+1}")
plt.legend()
plt.title("Before Grouping: Separate Univariate Series")
plt.xlabel("Time")
plt.ylabel("Target Value")
plt.show()

# ----------------------------
# 3. Make a GluonTS dataset and group
# ----------------------------
dataset = ListDataset(series_list, freq="D")
grouper = MultivariateGrouper()
multi_dataset = grouper(dataset)
entry = list(multi_dataset)[0]  # single grouped entry

print("Grouped target shape:", entry["target"].shape)
if "feat_dynamic_real" in entry:
    print("Grouped covariates shape:", entry["feat_dynamic_real"].shape)

# ----------------------------
# 4. Plot after grouping
# ----------------------------
plt.figure(figsize=(10, 5))
for i in range(entry["target"].shape[0]):
    plt.plot(time, entry["target"][i], label=f"Grouped Series {i+1}")
plt.legend()
plt.title("After Grouping: Multivariate Series (N×T)")
plt.xlabel("Time")
plt.ylabel("Target Value")
plt.show()

# ----------------------------
# 5. Plot grouped covariates (optional)
# ----------------------------
plt.figure(figsize=(10, 5))
for i in range(entry["feat_dynamic_real"].shape[0]):
    plt.plot(time, entry["feat_dynamic_real"][i], label=f"Covariate {i+1}")
plt.legend()
plt.title("Grouped Covariates")
plt.xlabel("Time")
plt.ylabel("Covariate Value")
plt.show()
