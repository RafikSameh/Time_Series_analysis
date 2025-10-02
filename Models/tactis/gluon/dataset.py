"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import logging
from gluonts.dataset.common import DataEntry, MetaData, Dataset, ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import datasets as monash_datasets
from gluonts.dataset.repository.datasets import (
    dataset_recipes,
    default_dataset_path,
    generate_forecasting_dataset,
    get_dataset,
    partial,
)
from gluonts.dataset.common import ListDataset
import pandas as pd
from functools import partial
from typing import List, Tuple, Optional


_DATA_BACKTEST_DEF = {
    "solar_10min": {  # 137 series, 72 prediction length
        "train_dates": [
            pd.Timestamp(year=2006, month=11, day=20),  # Monday
            pd.Timestamp(year=2006, month=11, day=27),
            pd.Timestamp(year=2006, month=12, day=4),
            pd.Timestamp(year=2006, month=12, day=11),
            pd.Timestamp(year=2006, month=12, day=18),
            pd.Timestamp(year=2006, month=12, day=25),
        ],
        "end_date": pd.Timestamp(year=2007, month=1, day=1),  # A Monday
    },
    "electricity_hourly": {  # 321 series, 24 prediction length
        "train_dates": [
            pd.Timestamp(year=2014, month=11, day=17),  # A Monday
            pd.Timestamp(year=2014, month=11, day=24),
            pd.Timestamp(year=2014, month=12, day=1),
            pd.Timestamp(year=2014, month=12, day=8),
            pd.Timestamp(year=2014, month=12, day=15),
            pd.Timestamp(year=2014, month=12, day=22),
        ],
        "end_date": pd.Timestamp(year=2014, month=12, day=29),  # Last Monday before end of data
    },
    "kdd_cup_2018_without_missing": {  # 270 series, 48 prediction length
        "train_dates": [
            pd.Timestamp(year=2018, month=1, day=1),  # A Monday
            pd.Timestamp(year=2018, month=1, day=15),
            pd.Timestamp(year=2018, month=1, day=29),
            pd.Timestamp(year=2018, month=2, day=12),
            pd.Timestamp(year=2018, month=2, day=26),
            pd.Timestamp(year=2018, month=3, day=12),
        ],
        "end_date": pd.Timestamp(year=2018, month=3, day=26),  # Last Monday before end of data
    },
    "traffic": {  # 862 series, 24 prediction length
        "train_dates": [
            pd.Timestamp(year=2016, month=11, day=14),  # A Monday
            pd.Timestamp(year=2016, month=11, day=21),
            pd.Timestamp(year=2016, month=11, day=28),
            pd.Timestamp(year=2016, month=12, day=5),
            pd.Timestamp(year=2016, month=12, day=12),
            pd.Timestamp(year=2016, month=12, day=19),
        ],
        "end_date": pd.Timestamp(year=2016, month=12, day=26),  # Last Monday before end of data
    },
    "fred_md": {  # 107 series, 12 prediction length
        "train_dates": [
            pd.Timestamp(year=2013, month=1, day=30),
            pd.Timestamp(year=2014, month=1, day=30),
            pd.Timestamp(year=2015, month=1, day=30),
            pd.Timestamp(year=2016, month=1, day=30),
            pd.Timestamp(year=2017, month=1, day=30),
            pd.Timestamp(year=2018, month=1, day=30),
        ],
        "end_date": pd.Timestamp(year=2019, month=1, day=30),  # Last January before end of data
    },
    "my_csv": {
        "train_dates": [
            pd.Timestamp("2013-11-01 00:00:00"),
            pd.Timestamp("2013-11-08 00:00:00"),
            pd.Timestamp("2013-11-15 00:00:00"),
            pd.Timestamp("2013-11-22 00:00:00"),
            pd.Timestamp("2013-11-29 00:00:00"),
            pd.Timestamp("2013-12-06 00:00:00"),
            pd.Timestamp("2013-12-13 00:00:00"),
            pd.Timestamp("2013-12-20 00:00:00"),
        ],
        "end_date": pd.Timestamp("2013-12-27 00:00:00"),
    },
}

_DATA_PREBACKTEST_DEF = {
    "solar_10min": {  # 137 series, 72 prediction length
        "train_dates": [
            pd.Timestamp(year=2006, month=11, day=13),  # Monday
            pd.Timestamp(year=2006, month=11, day=20),
            pd.Timestamp(year=2006, month=11, day=27),
            pd.Timestamp(year=2006, month=12, day=4),
            pd.Timestamp(year=2006, month=12, day=11),
            pd.Timestamp(year=2006, month=12, day=18),
        ],
        "end_date": pd.Timestamp(year=2006, month=12, day=25),  # A Monday
    },
    "kdd_cup_2018_without_missing": {  # 270 series, 48 prediction length
        "train_dates": [
            pd.Timestamp(year=2017, month=12, day=18),  # A Monday
            pd.Timestamp(year=2018, month=1, day=1),
            pd.Timestamp(year=2018, month=1, day=15),
            pd.Timestamp(year=2018, month=1, day=29),
            pd.Timestamp(year=2018, month=2, day=12),
            pd.Timestamp(year=2018, month=2, day=26),
        ],
        "end_date": pd.Timestamp(year=2018, month=3, day=12),  # Last Monday before end of data
    },
    "fred_md": {  # 107 series, 12 prediction length
        "train_dates": [
            pd.Timestamp(year=2012, month=1, day=30),
            pd.Timestamp(year=2013, month=1, day=30),
            pd.Timestamp(year=2014, month=1, day=30),
            pd.Timestamp(year=2015, month=1, day=30),
            pd.Timestamp(year=2016, month=1, day=30),
            pd.Timestamp(year=2017, month=1, day=30),
        ],
        "end_date": pd.Timestamp(year=2018, month=1, day=30),  # Last January before end of data
    },
    "electricity_hourly": {  # 107 series, 12 prediction length
        "train_dates": [
            pd.Timestamp(year=2014, month=11, day=10),  # A Monday
            pd.Timestamp(year=2014, month=11, day=17),  # A Monday
            pd.Timestamp(year=2014, month=11, day=24),
            pd.Timestamp(year=2014, month=12, day=1),
            pd.Timestamp(year=2014, month=12, day=8),
            pd.Timestamp(year=2014, month=12, day=15),
        ],
        "end_date": pd.Timestamp(year=2014, month=12, day=22),  # Last Monday before end of data
    },
    "traffic": {  # 862 series, 24 prediction length
        "train_dates": [
            pd.Timestamp(year=2016, month=11, day=7),  # A Monday
            pd.Timestamp(year=2016, month=11, day=14),  # A Monday
            pd.Timestamp(year=2016, month=11, day=21),
            pd.Timestamp(year=2016, month=11, day=28),
            pd.Timestamp(year=2016, month=12, day=5),
            pd.Timestamp(year=2016, month=12, day=12),
        ],
        "end_date": pd.Timestamp(year=2016, month=12, day=19),  # Last Monday before end of data
    },
    "my_csv": {
        "train_dates": [
            pd.Timestamp("2013-10-25 00:00:00"),
            pd.Timestamp("2013-11-01 00:00:00"),
            pd.Timestamp("2013-11-08 00:00:00"),
            pd.Timestamp("2013-11-15 00:00:00"),
            pd.Timestamp("2013-11-22 00:00:00"),
            pd.Timestamp("2013-11-29 00:00:00"),
        ],
        "end_date": pd.Timestamp("2013-12-06 00:00:00"),
    },
}


def _monash_inject_datasets(name, filename, record, prediction_length=None):
    """
    Injects datasets from the Monash Time Series Repository that were not included in GluonTS.
    """
    dataset_recipes.update(
        {
            name: partial(
                generate_forecasting_dataset,
                dataset_name=name,
                prediction_length=prediction_length,
            )
        }
    )
    monash_datasets.update({name: MonashDataset(file_name=filename, record=record)})


base_dir = os.path.dirname(os.path.abspath(__file__))
my_csv_folder_path = os.path.join(base_dir, "my_csv")

# Modifications to the GluonTS dataset repository
# * We add missing datasets from the Monash Time Series Repository
# * We rename datasets to have a prefix that makes their source explicit
_monash_inject_datasets(
    "electricity_hourly",
    "electricity_hourly_dataset.zip",
    "4656140",
    prediction_length=24,
)
_monash_inject_datasets(
    "my_csv",
    my_csv_folder_path,
    "custom_telecom_dataset_v1",
    prediction_length=24
)

_monash_inject_datasets("solar_10min", "solar_10_minutes_dataset.zip", "4656144", prediction_length=72)
_monash_inject_datasets("traffic", "traffic_hourly_dataset.zip", "4656132", prediction_length=24)


def _count_timesteps(left: pd.Timestamp, right: pd.Timestamp, delta: pd.Timedelta) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    """

    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0.
    # Original code was tested on version 0.9.4
    if isinstance(left, pd.Period):
        left = left.to_timestamp()
    if isinstance(right, pd.Period):
        right = right.to_timestamp()

    assert right >= left, f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."

    # Use Timedelta for division, as it's the correct way to handle time differences.
    return int((right - left) // delta)


#################################################### load_custom_dataset ###################################################################################

def load_custom_dataset_folder(
    folder_path: str,
    freq: str = "1H",
    prediction_length: int = 6,
    use_cached: bool = True
):
    print("✅ Entered load_custom_dataset_folder")

    series_list = []
    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️ Skipping empty file: {file}")
                continue
        except Exception as e:
            print(f"⚠️ Skipping broken file: {file} ({e})")
            continue

        # --- check for required columns ---
        if "internet" not in df.columns:
            print(f"⚠️ Skipping file (no 'internet' column): {file}")
            continue
        if "datetime" not in df.columns:
            print(f"⚠️ Skipping file (no 'datetime' column): {file}")
            continue

        # --- extract target ---
        target = df["internet"].to_numpy(dtype=np.float32)

        # ✅ ensure target always has shape (features, time)
        if target.ndim == 1:
            target = target[None, :]   # (time,) -> (1, time)

        # --- covariates (dynamic features) ---
        # --- covariates (dynamic features) ---
        drop_cols = ["internet", "datetime","day_of_week","is_weekend","month","internet_roll_mean_24h","internet_roll_std_24h","internet_lag_24h"]
        covariates = df.drop(columns=[c for c in drop_cols if c in df.columns])

        feat_dynamic_real = covariates.to_numpy(dtype=np.float32).T  # (features, time)

        start = pd.to_datetime(df["datetime"].iloc[0])

        series = {
            "target": target,                 # always (C, T)
            "start": start,
            "feat_dynamic_real": feat_dynamic_real,
            "item_id": file.split(".")[0],
        }
        series_list.append(series)

        print(f"📂 File: {file}, Item ID: {series['item_id']}")
        print(f"Target shape: {series['target'].shape}, "
              f"first 5 values: {series['target'][0, :5]}")
        print(f"Covariates shape: {feat_dynamic_real.shape} (features x time)")
        print(f"Covariate columns: {covariates.columns.tolist()}")
        print(f"First feature (first 5 values): {feat_dynamic_real[0, :5]}")
        print()

    print(f"✅ Finished loading {len(series_list)} series from {folder_path}")
    if len(series_list) > 0:
        print("Target shape:", series_list[0]["target"].shape)
        print("Dynamic feature shape:", series_list[0]["feat_dynamic_real"].shape)

    # return Meta + dataset
    class Meta:
        def __init__(self, freq, prediction_length):
            self.freq = freq
            self.prediction_length = prediction_length

    return Meta(freq, prediction_length), series_list



#######################################################################################################################################


def _load_raw_dataset(name: str, use_cached: bool) -> Tuple[MetaData, List[DataEntry]]:
    """
    Load the dataset using GluonTS method, and combining both the train and test data.
    """
    if name == "my_csv":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(base_dir, "my_csv")

        print(f"📂 Loading data from folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ Folder not found: {folder_path}")
        
        metadata, raw_dataset = load_custom_dataset_folder(
            folder_path=folder_path,
            freq="1H",
            prediction_length=24,
            use_cached=use_cached
        )
        
        if not raw_dataset or len(raw_dataset) == 0:
            raise ValueError("⚠️ Loaded dataset is empty!")
        
        #first_sample = raw_dataset[1]
        #print(f"🔑 First sample keys: {list(first_sample.keys())}")

        first_sample = raw_dataset[0]  # safer
        print(f"Target shape: {first_sample['target'].shape}")
        if 'feat_dynamic_real' in first_sample:
            print(f"Dynamic feature shape: {first_sample['feat_dynamic_real'].shape}")
        
        if 'target' not in first_sample:
            raise KeyError("❌ Dataset sample missing 'target' field!")
        
        target = first_sample['target']
        if not isinstance(target, (np.ndarray, list)):
            raise TypeError(f"❌ 'target' field should be np.ndarray or list, but got {type(target)}")
        
        if len(target) == 0:
            raise ValueError("⚠️ The 'target' array in the first sample is empty!")
        
        print("✅ Dataset loaded and validated successfully!")
        
        return metadata, raw_dataset


    cache_path = Path(os.environ.get("TACTIS_DATA_STORE", default_dataset_path))
    uv_dataset = get_dataset(name, regenerate=not use_cached, path=cache_path)

    timestep_delta = pd.to_timedelta(uv_dataset.metadata.freq)

    data = [series.copy() for series in uv_dataset.train]

    for i, new_series in enumerate(uv_dataset.test):
        old_series = data[i % len(data)]

        if "feat_static_cat" in new_series:
            assert old_series["feat_static_cat"] == new_series["feat_static_cat"]

        if old_series["start"] > new_series["start"]:
            extra_timesteps = _count_timesteps(new_series["start"], old_series["start"], timestep_delta)
            assert old_series["target"][0] == new_series["target"][extra_timesteps]
            old_series["start"] = new_series["start"]
            old_series["target"] = np.concatenate([new_series["target"][0:extra_timesteps], old_series["target"]])

        old_end = old_series["start"] + len(old_series["target"]) * timestep_delta
        new_end = new_series["start"] + len(new_series["target"]) * timestep_delta
        if new_end > old_end:
            extra_timesteps = _count_timesteps(old_end, new_end, timestep_delta)
            assert old_series["target"][-1] == new_series["target"][-extra_timesteps - 1]
            old_series["target"] = np.concatenate([old_series["target"], new_series["target"][-extra_timesteps:]])

    return uv_dataset.metadata, data

def generate_hp_search_datasets(
    name: str,
    history_length_multiple: float,
    use_cached: bool = True,
    validation_series_limit: Optional[int] = None,
) -> Tuple[MetaData, Dataset, Dataset]:
    """
    Generate the training and validation datasets to be used during the hyperparameter search.
    """
    print("eneterd generate_hp_search_datasetsssssssss")
    metadata, raw_dataset = _load_raw_dataset(name, use_cached=use_cached)
    print("metdataaaaaaaaaaaaaaaaaaaaaaaa = ", metadata)

    # fixed split index (assumes each series is long enough)
    split_point_index = 1200

    train_data = []
    valid_data = []
    for i, series in enumerate(raw_dataset):
        tgt = series["target"]
        feat_dyn = series.get("feat_dynamic_real", None)

        # --- train slice ---
        s_train = series.copy()
        if tgt.ndim == 1:
            s_train["target"] = tgt[:split_point_index]
        else:
            s_train["target"] = tgt[:, :split_point_index]

        if isinstance(feat_dyn, np.ndarray):
            # shape: (num_features, T) -> slice time dimension
            s_train["feat_dynamic_real"] = feat_dyn[:, :split_point_index]

        s_train["item_id"] = i
        train_data.append(s_train)

        # --- valid slice ---
        s_valid = series.copy()
        s_valid["start"] = s_valid["start"] + split_point_index * pd.to_timedelta(metadata.freq)
        if tgt.ndim == 1:
            s_valid["target"] = tgt[split_point_index:]
        else:
            s_valid["target"] = tgt[:, split_point_index:]

        if isinstance(feat_dyn, np.ndarray):
            s_valid["feat_dynamic_real"] = feat_dyn[:, split_point_index:]

        valid_data.append(s_valid)

    if validation_series_limit is not None and validation_series_limit < len(valid_data):
        indices = np.random.choice(len(valid_data), size=validation_series_limit, replace=False)
        valid_data = [valid_data[i] for i in sorted(indices)]

    # ---- Debug prints (handle 1D/2D) ----
    print("------------------------------")
    print("✅ Data split into training and validation sets.")
    print(f"➡️ Training data contains {len(train_data)} series.")
    print(f"➡️ Validation data contains {len(valid_data)} series.")

    if len(train_data) > 0:
        ts = train_data[0]["target"]
        print(f"First training series 'target' shape: {ts.shape}")
        if ts.ndim == 1:
            print(f"First training series 'target' first 5 values: {ts[:5]}")
        else:
            print(f"First training series 'target' first 5 values (row 0): {ts[0, :5]}")
        if "feat_dynamic_real" in train_data[0]:
            print(f"First training series feat_dynamic_real shape: {train_data[0]['feat_dynamic_real'].shape}")

    if len(valid_data) > 0:
        ts = valid_data[0]["target"]
        print(f"First validation series 'target' shape: {ts.shape}")
        if ts.ndim == 1:
            print(f"First validation series 'target' first 5 values: {ts[:5]}")
        else:
            print(f"First validation series 'target' first 5 values (row 0): {ts[0, :5]}")
        if "feat_dynamic_real" in valid_data[0]:
            print(f"First validation series feat_dynamic_real shape: {valid_data[0]['feat_dynamic_real'].shape}")
    print("------------------------------")

    # Choose one_dim_target dynamically (True for univariate, False for multivariate)
    one_dim = (train_data[0]["target"].ndim == 1)

    train_ds = ListDataset(train_data, freq=metadata.freq, one_dim_target=one_dim)
    valid_ds = ListDataset(valid_data, freq=metadata.freq, one_dim_target=one_dim)

    print(f"Number of training time series: {len(train_data)}")
    print(f"Number of validation time series: {len(valid_data)}")

    first_train_series = next(iter(train_data))
    first_valid_series = next(iter(valid_data))
    print(f"First training series starts at: {first_train_series['start']}")
    print(f"Shape of first training series target: {first_train_series['target'].shape}")
    print(f"First validation series starts at: {first_valid_series['start']}")
    print(f"Shape of first validation series target: {first_valid_series['target'].shape}")

    return metadata, train_ds, valid_ds


def maximum_backtest_id(name: str) -> int:
    """
    Return the largest possible backtesting id for the given dataset.
    """
    return len(_DATA_BACKTEST_DEF[name]["train_dates"])


class __FixedMultivariateGrouper(MultivariateGrouper):
    """
    Temporary fix for MultivariateGrouper when used with NumPy >= 1.24.
    See: https://github.com/awslabs/gluonts/issues/2612
    """
    print("enetered __FixedMultivariateGrouper ")

    def _prepare_test_data(self, dataset: Dataset) -> Dataset:
        assert self.num_test_dates is not None
        assert len(dataset) % self.num_test_dates == 0

        logging.info("group test time series to datasets")

        test_length = len(dataset) // self.num_test_dates

        all_entries = list()
        for test_start in range(0, len(dataset), test_length):
            dataset_at_test_date = dataset[test_start : test_start + test_length]
            transformed_target = self._transform_target(self._left_pad_data, dataset_at_test_date)[FieldName.TARGET]

            grouped_data = dict()
            grouped_data[FieldName.TARGET] = np.array(list(transformed_target), dtype=np.float32)
            for data in dataset:
                fields = data.keys()
                break
            if FieldName.FEAT_DYNAMIC_REAL in fields:
                grouped_data[FieldName.FEAT_DYNAMIC_REAL] = np.vstack(
                    [data[FieldName.FEAT_DYNAMIC_REAL] for data in dataset],
                )
            grouped_data = self._restrict_max_dimensionality(grouped_data)
            grouped_data[FieldName.START] = self.first_timestamp
            grouped_data[FieldName.FEAT_STATIC_CAT] = [0]
            all_entries.append(grouped_data)

        return ListDataset(all_entries, freq=self.frequency, one_dim_target=False)


def generate_prebacktesting_datasets(
    name: str, backtest_id: int, history_length_multiple: float, use_cached: bool = True
) -> Tuple[MetaData, Dataset, Dataset]:
    """
    Generate the training and testing datasets to be used during the backtesting.
    """
    print("enetered generate_prebacktesting_datasets")
    metadata, raw_dataset = _load_raw_dataset(name, use_cached=use_cached)

    backtest_timestamp = _DATA_PREBACKTEST_DEF[name]["train_dates"][backtest_id]
    history_length = int(history_length_multiple * metadata.prediction_length)

    timestep_delta = pd.to_timedelta(metadata.freq)
    test_offset = timestep_delta * metadata.prediction_length
    if backtest_id + 1 < maximum_backtest_id(name):
        num_test_dates = _count_timesteps(
            _DATA_PREBACKTEST_DEF[name]["train_dates"][backtest_id],
            _DATA_PREBACKTEST_DEF[name]["train_dates"][backtest_id + 1],
            test_offset,
        )
    else:
        num_test_dates = _count_timesteps(
            _DATA_PREBACKTEST_DEF[name]["train_dates"][backtest_id],
            _DATA_PREBACKTEST_DEF[name]["end_date"],
            test_offset,
        )

    train_data = []
    for i, series in enumerate(raw_dataset):
        tgt = series["target"]
        feat_dyn = series.get("feat_dynamic_real", None)
        train_end_index = _count_timesteps(series["start"], backtest_timestamp, timestep_delta)

        s_train = series.copy()
        s_train["target"] = tgt[:train_end_index]
        if isinstance(feat_dyn, np.ndarray):
            s_train["feat_dynamic_real"] = feat_dyn[:, :train_end_index]
        s_train["item_id"] = i
        train_data.append(s_train)

    test_data = []
    for test_id in range(num_test_dates):
        for i, series in enumerate(raw_dataset):
            tgt = series["target"]
            feat_dyn = series.get("feat_dynamic_real", None)

            train_end_index = _count_timesteps(series["start"], backtest_timestamp, timestep_delta)
            test_end_index = train_end_index + metadata.prediction_length * (test_id + 1)
            test_start_index = test_end_index - metadata.prediction_length - history_length

            s_test = series.copy()
            s_test["start"] = series["start"] + test_start_index * timestep_delta
            s_test["target"] = tgt[test_start_index:test_end_index]
            if isinstance(feat_dyn, np.ndarray):
                s_test["feat_dynamic_real"] = feat_dyn[:, test_start_index:test_end_index]
            s_test["item_id"] = len(test_data)
            test_data.append(s_test)

    #train_grouper = MultivariateGrouper()
    #test_grouper = __FixedMultivariateGrouper(num_test_dates=num_test_dates)

    train_grouper = ListDataset(train_data, freq=metadata.freq, one_dim_target=False)
    test_grouper  = ListDataset(test_data, freq=metadata.freq, one_dim_target=False)

    return metadata, train_grouper(train_data), test_grouper(test_data)

def generate_backtesting_datasets(
    name: str, backtest_id: int, history_length_multiple: float, use_cached: bool = True
) -> Tuple[MetaData, Dataset, Dataset]:
    """
    Generate the training and testing datasets to be used during the backtesting.
    """
    print("eneterdddddddddddddd generate_backtesting_datasets")
    metadata, raw_dataset = _load_raw_dataset(name, use_cached=use_cached)

    backtest_timestamp = _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id]
    history_length = int(history_length_multiple * metadata.prediction_length)

    timestep_delta = pd.to_timedelta(metadata.freq)
    test_offset = timestep_delta * metadata.prediction_length
    if backtest_id + 1 < maximum_backtest_id(name):
        num_test_dates = _count_timesteps(
            _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id],
            _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id + 1],
            test_offset,
        )
    else:
        num_test_dates = _count_timesteps(
            _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id],
            _DATA_BACKTEST_DEF[name]["end_date"],
            test_offset,
        )

    train_data = []
    for i, series in enumerate(raw_dataset):
        tgt = series["target"]
        feat_dyn = series.get("feat_dynamic_real", None)
        train_end_index = _count_timesteps(series["start"], backtest_timestamp, timestep_delta)

        s_train = series.copy()
        s_train["target"] = tgt[:train_end_index]
        if isinstance(feat_dyn, np.ndarray):
            s_train["feat_dynamic_real"] = feat_dyn[:, :train_end_index]
        # s_train["item_id"] = i
        train_data.append(s_train)

    test_data = []
    for test_id in range(num_test_dates):
        for i, series in enumerate(raw_dataset):
            tgt = series["target"]
            feat_dyn = series.get("feat_dynamic_real", None)

            train_end_index = _count_timesteps(series["start"], backtest_timestamp, timestep_delta)
            test_end_index = train_end_index + metadata.prediction_length * (test_id + 1)
            test_start_index = test_end_index - metadata.prediction_length - history_length

            s_test = series.copy()
            s_test["start"] = series["start"] + test_start_index * timestep_delta
            s_test["target"] = tgt[test_start_index:test_end_index]
            if isinstance(feat_dyn, np.ndarray):
                s_test["feat_dynamic_real"] = feat_dyn[:, test_start_index:test_end_index]
            s_test["item_id"] = len(test_data)
            test_data.append(s_test)

    #train_grouper = MultivariateGrouper()
    #test_grouper = __FixedMultivariateGrouper(num_test_dates=num_test_dates)

    train_grouper = ListDataset(train_data, freq=metadata.freq, one_dim_target=False)
    test_grouper  = ListDataset(test_data, freq=metadata.freq, one_dim_target=False)

    return metadata, train_grouper(train_data), test_grouper(test_data)