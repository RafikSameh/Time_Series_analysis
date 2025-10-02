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

# Ignore warnings to get a clean output
import warnings
import os
import json
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
from gluonts.dataset.repository.datasets import get_dataset
import numpy as np
import pandas as pd

import torch
import argparse
import matplotlib.pyplot as plt

from Models.tactis.gluon.utils import set_seed
from Models.tactis.gluon.estimator import TACTiSEstimator
from Models.tactis.gluon.trainer import TACTISTrainer
from Models.tactis.gluon.dataset import (
    generate_hp_search_datasets,
    generate_backtesting_datasets,
    generate_prebacktesting_datasets,
    load_custom_dataset_folder,
)
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from Models.tactis.model.utils import check_memory
from Models.tactis.gluon.metrics import compute_validation_metrics, compute_validation_metrics_interpolation


print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ Current device ID:", torch.cuda.current_device())
    print("✅ Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("✅ Running on CPU")




# Moved these dictionaries to the global scope so they can be imported
series_length_maps = {
    "solar_10min": 137,
    "electricity_hourly": 321,
    "kdd_cup_2018_without_missing": 270,
    "traffic": 862,
    "fred_md": 107,
    "my_csv": 1488,  # number of samples per series
}

prediction_length_maps = {
    "solar_10min": 72,
    "electricity_hourly": 24,
    "kdd_cup_2018_without_missing": 48,
    "traffic": 24,
    "fred_md": 12,
    "my_csv": 24,  # Correct key and value for prediction length
}

prediction_length_maps_freq = {
    "my_csv": "1H",  # New dictionary for frequency
}

model_parameters_maps = {
    "my_csv": {
        "flow_temporal_encoder": {
            "attention_layers": 1,
            "attention_heads": 4,
            "attention_dim": 24,
            "attention_feedforward_dim": 24,
            "dropout": 0.0,
        },
        "copula_temporal_encoder": {
            "attention_layers": 1,
            "attention_heads": 4,
            "attention_dim": 24,
            "attention_feedforward_dim": 24,
            "dropout": 0.0,
        },
        "flow_series_embedding_dim": 24,
        "copula_series_embedding_dim": 24,
        "flow_input_encoder_layers": 1,
        "copula_input_encoder_layers": 1,
        "bagging_size": 20,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": True,
        "positional_encoding": {
            "dropout": 0.0,
        },
        "copula_decoder": {
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                "attention_heads": 4,
                "attention_layers": 1,
                "attention_dim": 24,
                "mlp_layers": 1,
                "mlp_dim": 32,
                "resolution": 16,
                "attention_mlp_class": "MLP",
                "dropout": 0.0,
                "activation_function": "relu",
            },
            "dsf_marginal": {
                "mlp_layers": 1,
                "mlp_dim": 24,
                "flow_layers": 1,
                "flow_hid_dim": 24,
            },
        },
        "experiment_mode": "forecasting",
        "skip_copula": True,
    }
}






def main(args):
    seed = args.seed
    num_workers = args.num_workers
    history_factor = args.history_factor
    epochs = args.epochs
    load_checkpoint = args.load_checkpoint
    activation_function = args.decoder_act
    dataset = args.dataset
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    clip_gradient = args.clip_gradient
    checkpoint_dir = args.checkpoint_dir

    if args.use_cpu:
        print("Using CPU")

    # Print memory avl.
    if not args.use_cpu:
        total, used = check_memory(0)
        print("Used/Total GPU memory:", used, "/", total)

    # Restrict memory to 12 GB if it greater than 12 GB
    # 12198 is the exact memory of a 12 GB P100
    if not args.do_not_restrict_memory and not args.use_cpu:
        if int(total) > 12198:
            fraction_to_use = 11598 / int(total)
            torch.cuda.set_per_process_memory_fraction(fraction_to_use, 0)
            print("Restricted memory to 12 GB")

    if args.evaluate:
        # Bagging is disabled during evaluation
        args.bagging_size = None
        # Assert that there is a checkpoint to evaluate
        assert load_checkpoint, "Please set --load_checkpoint for evaluation"
        # Get the stage of the model to evaluate
        stage = torch.load(load_checkpoint)["stage"]
        skip_copula = stage == 1
    else:
        skip_copula = True

    series_length_maps = {
        "solar_10min": 137,
        "electricity_hourly": 321,
        "kdd_cup_2018_without_missing": 270,
        "traffic": 862,
        "fred_md": 107,
        "my_csv": 1488,  # number of samples per series
    }

    prediction_length_maps = {
        "solar_10min": 72,
        "electricity_hourly": 24,
        "kdd_cup_2018_without_missing": 48,
        "traffic": 24,
        "fred_md": 12,
        "my_csv": 24,  # Correct key and value for prediction length
    }

    prediction_length_maps_freq = {
        "my_csv": "1H",  # New dictionary for frequency
    }
    
    ### Decide the prediction factor for the dataloader
    prediction_length = prediction_length_maps[dataset]
    print("Using history factor:", history_factor)
    print("Prediction length of the dataset:", prediction_length_maps[dataset])


    # If it is evaluation for interpolation, we use a trick to perform interpolation with GluonTS
    # Increase history factor by 1, and get the interpolation prediction window from the history window itself
    # This may be refactored later if we remove the GluonTS dependency for the sample() functions for interpolation
    if args.experiment_mode == "interpolation" and args.evaluate:
        history_factor += 1

    if args.bagging_size:
        assert args.bagging_size < series_length_maps[dataset]

    encoder_dict = {
        "flow_temporal_encoder": {
            "attention_layers": args.flow_encoder_num_layers,
            "attention_heads": args.flow_encoder_num_heads,
            "attention_dim": args.flow_encoder_dim,
            "attention_feedforward_dim": args.flow_encoder_dim,
            "dropout": 0.0,
        },
        "copula_temporal_encoder": {
            "attention_layers": args.copula_encoder_num_layers,
            "attention_heads": args.copula_encoder_num_heads,
            "attention_dim": args.copula_encoder_dim,
            "attention_feedforward_dim": args.copula_encoder_dim,
            "dropout": 0.0,
        },
    }

    # num_series is passed separately by the estimator
    model_parameters = {
        "flow_series_embedding_dim": args.flow_series_embedding_dim,
        "copula_series_embedding_dim": args.copula_series_embedding_dim,
        "flow_input_encoder_layers": args.flow_input_encoder_layers,
        "copula_input_encoder_layers": args.copula_input_encoder_layers,
        "bagging_size": args.bagging_size,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": args.loss_normalization,
        "positional_encoding": {
            "dropout": 0.0,
        },
        **encoder_dict,
        "copula_decoder": {
            # flow_input_dim and copula_input_dim are passed by the TACTIS module dynamically
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                "attention_heads": args.decoder_num_heads,
                "attention_layers": args.decoder_num_layers,
                "attention_dim": args.decoder_dim,
                "mlp_layers": args.decoder_mlp_layers,
                "mlp_dim": args.decoder_mlp_dim,
                "resolution": args.decoder_resolution,
                "attention_mlp_class": args.decoder_attention_mlp_class,
                "dropout": 0.0,
                "activation_function": activation_function,
            },
            "dsf_marginal": {
                "mlp_layers": args.dsf_mlp_layers,
                "mlp_dim": args.dsf_mlp_dim,
                "flow_layers": args.dsf_num_layers,
                "flow_hid_dim": args.dsf_dim,
            },
        },
        "experiment_mode": args.experiment_mode,
        "skip_copula": skip_copula,
    }

    set_seed(seed)
    if args.backtest_id >= 0 and args.backtest_id <= 5:
        backtesting = True
        print("Using backtest dataset with ID", args.backtest_id)
        if not args.prebacktest:
            print("CAUTION: The validation set here is the actual test set.")
            metadata, train_data, valid_data = generate_backtesting_datasets(dataset, args.backtest_id, history_factor)
        else:
            print("Using the prebacktesting set.")
            backtesting = False
            metadata, train_data, valid_data = generate_prebacktesting_datasets(
                dataset, args.backtest_id, history_factor
            )
            _, _, test_data = generate_backtesting_datasets(dataset, args.backtest_id, history_factor)
    else:
        backtesting = False
        print("Using HP search dataset")
        metadata, train_data, valid_data = generate_hp_search_datasets(dataset, history_factor, validation_series_limit=args.validation_series_limit,)
######################################################################################
        print("✅ Plotting first few time series from train_data...")
        N = 5
        for i, entry in enumerate(train_data):
            target = entry["target"]
            start = entry["start"]

            plt.figure(figsize=(10, 6))

            if target.ndim == 1:  # univariate
                plt.plot(target, label="Series 1")
            else:  # multivariate
                for series_idx in range(target.shape[0]):
                    plt.plot(target[series_idx], label=f"Series {series_idx+1}")

            plt.title(f"Train Series #{i} (starts at {start})")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"train_series_{i}.png")

            if i + 1 >= N:
                break

        print(f"✅ Saved {N} plots as train_series_0.png to train_series_{N-1}.png")

######################################################################################
    num_features = 1
    set_seed(seed)
    history_length = history_factor * metadata.prediction_length
    estimator_custom = TACTiSEstimator(
        model_parameters=model_parameters,
       # num_series=train_data.list_data[0]["target"].shape[0],
        num_series=len(train_data) * train_data[0]["target"].shape[0],

        history_length=history_length,
        prediction_length=prediction_length,
        freq=metadata.freq,
        trainer=TACTISTrainer(
            epochs=epochs,
            batch_size=args.batch_size,
            training_num_batches_per_epoch=args.training_num_batches_per_epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clip_gradient=clip_gradient,
            device=torch.device("cuda") if not args.use_cpu else torch.device("cpu"),
            log_subparams_every=args.log_subparams_every,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            load_checkpoint=load_checkpoint,
            early_stopping_epochs=args.early_stopping_epochs,
            do_not_restrict_time=args.do_not_restrict_time,
            skip_batch_size_search=args.skip_batch_size_search,
            prediction_length=prediction_length,
            num_features=num_features,
        ),
        cdf_normalization=False,
        num_parallel_samples=100,
    )
    print("✅ Trainer is using device:", "cuda" if not args.use_cpu else "cpu")

     # Corrected logical flow for training, evaluation, and forecasting
    if not args.evaluate and not args.forecast:
        # This is the main training logic
        print("Checkpoints will be saved at", checkpoint_dir)
        #print("model_parametersssssssssssssss  = ", estimator_custom.model_kwargs)
        
        # Training
        estimator_custom.train(
            train_data,
            valid_data,
            num_workers=num_workers,
            optimizer=args.optimizer,
            backtesting=backtesting,
        )
    elif args.evaluate:
        # This is the evaluation logic
        transformation = estimator_custom.create_transformation()
        device = estimator_custom.trainer.device
        model = estimator_custom.create_training_network(device)
        print("✅ Model is on device:", next(model.parameters()).device)
        model_state_dict = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(model_state_dict["model"])

        predictor_custom = estimator_custom.create_predictor(
            transformation=transformation,
            trained_network=model,
            device=device,
            experiment_mode=args.experiment_mode,
            history_length=estimator_custom.history_length,
        )
        predictor_custom.freq = metadata.freq
        
        print("✅ Generating and plotting predictions...")
        forecast_it = predictor_custom.predict(
            valid_data, 
            num_samples=1000,
        )
        all_forecasts = list(forecast_it)
        prediction_length = estimator_custom.prediction_length
        # Select files and features to plot
        file_indices = []
        if args.evaluate_item_id != -1:
            try:
                # normalize both to string so it matches "4018" or "square4018"
                target_id = str(args.evaluate_item_id)
                single_file_index = next(
                    i for i, d in enumerate(valid_data) 
                    if str(d["item_id"]) == target_id or str(d["item_id"]) == f"square{target_id}"
                )
                file_indices = [single_file_index]
            except StopIteration:
                raise ValueError(f"File with ID {args.evaluate_item_id} not found in validation data.")
        else:
            file_indices = [10, 50, 100, 250, 400]

        parameter_indices = [0]
        fig, axes = plt.subplots(len(file_indices), len(parameter_indices), figsize=(20, 12), sharex=False)

        if len(file_indices) == 1:
            if len(parameter_indices) == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)

        for i, file_index in enumerate(file_indices):
            ax = axes[i, 0] if axes.ndim > 1 else axes[i]

            # target is (1, T), squeeze to (T,)
            full_actual_data = valid_data[file_index]["target"].squeeze(0)

            full_time_steps = np.arange(len(full_actual_data))
            forecast = all_forecasts[file_index]

            print("\n🔎 DEBUG INFO for file", valid_data[file_index]["item_id"])
            print("   forecast.mean shape:", forecast.mean.shape)
            print("   forecast.quantile(0.5) shape:", forecast.quantile(0.5).shape)

            # squeeze forecast outputs
            predicted_mean = forecast.mean.squeeze(-1)
            q05 = forecast.quantile(0.05).squeeze(-1)
            q95 = forecast.quantile(0.95).squeeze(-1)
            q25 = forecast.quantile(0.25).squeeze(-1)
            q75 = forecast.quantile(0.75).squeeze(-1)

            history_actual = full_actual_data[:-prediction_length]
            future_actual = full_actual_data[-prediction_length:]

            ax.plot(full_time_steps[:-prediction_length], history_actual, label="History", color='blue')
            ax.plot(full_time_steps[-prediction_length:], predicted_mean, label="Predicted", color='red', linestyle='--')
            ax.plot(full_time_steps[-prediction_length:], future_actual, label="Ground Truth", color='green')

            ax.fill_between(full_time_steps[-prediction_length:], q05, q95, color='red', alpha=0.1)
            ax.fill_between(full_time_steps[-prediction_length:], q25, q75, color='red', alpha=0.2)

            if i == 0:
                ax.set_title("Internet traffic")
            ax.set_ylabel(f"File {valid_data[file_index]['item_id']}")
            ax.grid(True)

        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        print("✅ Multi-file, multi-feature plots generated.")

    
    elif args.forecast:
        # This is the evaluation logic
        transformation = estimator_custom.create_transformation()
        device = estimator_custom.trainer.device
        model = estimator_custom.create_training_network(device)
        print("✅ Model is on device:", next(model.parameters()).device)
        model_state_dict = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(model_state_dict["model"])

        predictor_custom = estimator_custom.create_predictor(
            transformation=transformation,
            trained_network=model,
            device=device,
            experiment_mode=args.experiment_mode,
            history_length=estimator_custom.history_length,
        )
        predictor_custom.freq = metadata.freq

        print("✅ Generating and plotting predictions...")
        history_length = estimator_custom.history_length
        prediction_length = estimator_custom.prediction_length

        # 🖼️ Store original full data before any modifications
        full_actual_data_dict = {
            i: valid_data[i]["target"] for i in range(len(valid_data))
        }

        # Prepare mini_valid_data containing only the last history_length points
        mini_valid_data = []
        for i, series in enumerate(valid_data):
            mini_series = series.copy()
            mini_series["target"] = series["target"][:, -history_length:]  # last history_length points only
            mini_series["item_id"] = i
            mini_valid_data.append(mini_series)

        # 🔄 Rolling forecast: store each window in rolling_forecasts

        #rolling_steps = 3  # number of future windows to forecast
        rolling_steps = int(args.forecast_horizon / 24)
        print("🔄🔄🔄🔄rolling_steps = ", rolling_steps)
        rolling_forecasts = [[] for _ in mini_valid_data]  # list of lists

        for step in range(rolling_steps):
            forecast_it = predictor_custom.predict(
                mini_valid_data,
                num_samples=1000,
            )
            all_forecasts = list(forecast_it)

            for idx, forecast in enumerate(all_forecasts):
                mean_forecast = forecast.mean  # shape (prediction_length, num_features)
                rolling_forecasts[idx].append(forecast)  # store full forecast (mean + quantiles)

                # append predicted window to mini_valid_data for next step
                mini_valid_data[idx]["target"] = np.concatenate(
                    [mini_valid_data[idx]["target"], mean_forecast.T], axis=1
                )
                # keep only last history_length points for next prediction
                mini_valid_data[idx]["target"] = mini_valid_data[idx]["target"][:, -history_length:]

        # Select files and features to plot
        file_indices = []
        if args.evaluate_item_id != -1:
            try:
                # normalize both to string so it matches "4018" or "square4018"
                target_id = str(args.evaluate_item_id)
                single_file_index = next(
                    i for i, d in enumerate(valid_data) 
                    if str(d["item_id"]) == target_id or str(d["item_id"]) == f"square{target_id}"
                )
                file_indices = [single_file_index]
            except StopIteration:
                raise ValueError(f"File with ID {args.evaluate_item_id} not found in validation data.")
        else:
            file_indices = [10, 50, 100, 250, 400]


        parameter_indices = [0]
        
        fig, axes = plt.subplots(len(file_indices), len(parameter_indices), figsize=(20, 12), sharex=False)

        if len(file_indices) == 1:
            if len(parameter_indices) == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)

        colors = ['red', 'orange', 'purple', 'green', 'brown']  # colors for rolling windows

        for i, file_index in enumerate(file_indices):
            for j, parameter_index in enumerate(parameter_indices):
                ax = axes[i, j]

                # ✅ squeeze target to 1D
                full_actual_data = valid_data[file_index]["target"].squeeze(0)
                full_time_steps = np.arange(len(full_actual_data))

                # plot full actual data
                ax.plot(full_time_steps, full_actual_data, label="Actual Data", color='blue')

                # plot each rolling forecast window
                last_time_step = full_time_steps[-1]
                for step_idx, forecast_window in enumerate(rolling_forecasts[file_index]):
                    # ✅ squeeze forecasts to 1D
                    predicted_mean = forecast_window.mean.squeeze(-1)
                    q05 = forecast_window.quantile(0.05).squeeze(-1)
                    q95 = forecast_window.quantile(0.95).squeeze(-1)
                    q25 = forecast_window.quantile(0.25).squeeze(-1)
                    q75 = forecast_window.quantile(0.75).squeeze(-1)

                    pred_time_steps = np.arange(
                        last_time_step + 1 + step_idx * prediction_length,
                        last_time_step + 1 + (step_idx + 1) * prediction_length
                    )
                    color = colors[step_idx % len(colors)]

                    ax.plot(pred_time_steps, predicted_mean,
                            label=f"Predicted step {step_idx+1}", color=color, linestyle='--')
                    ax.fill_between(pred_time_steps, q05, q95, color=color, alpha=0.1)
                    ax.fill_between(pred_time_steps, q25, q75, color=color, alpha=0.2)

                if i == 0:
                    ax.set_title(f"Feature {parameter_index}")
                ax.set_ylabel(f"File {valid_data[file_index]['item_id']}")
                ax.grid(True)


        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # 🖼️ NEW CODE TO SAVE THE PLOT AS AN IMAGE
        plt.savefig("final_forecast.png", bbox_inches='tight')
        plt.show()

        print("✅ Multi-file, multi-feature rolling forecasts generated.")

        forecast_results = {}

        for i, file_index in enumerate(file_indices):
            file_id = str(valid_data[file_index]["item_id"])
            forecast_results[file_id] = {
                # 🖼️ Use the original, unmodified data
                "actual": full_actual_data_dict[file_index].tolist(),
                "forecasts": []
            }

            for step_idx, forecast_window in enumerate(rolling_forecasts[file_index]):
                forecast_entry = {
                    "step": step_idx + 1,
                    "mean": forecast_window.mean.tolist(),
                    "q05": forecast_window.quantile(0.05).tolist(),
                    "q25": forecast_window.quantile(0.25).tolist(),
                    "q75": forecast_window.quantile(0.75).tolist(),
                    "q95": forecast_window.quantile(0.95).tolist(),
                }
                forecast_results[file_id]["forecasts"].append(forecast_entry)
        
        print("sveddddddddddddddddddddddddddd")
        # Save everything into one JSON file
        with open("forecast_data.json", "w") as f:
            json.dump(forecast_results, f)

        
    else:
        # Default case for when neither --evaluate nor --forecast is provided
        pass # This block is now handled by the `if not args.evaluate and not args.forecast:` block above


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of multiprocessing workers.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size.")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs.")
    parser.add_argument(
        "--evaluate_item_id",
        type=int,
        default=-1,
        help="When evaluating, a specific item_id (e.g., 4002) to plot. -1 for all items.",
    )
    parser.add_argument("--forecast", action="store_true", help="Forecast the future beyond the last time step.")
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=24,
        help="Total number of steps to forecast into the future."
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["rmsprop", "adam"], help="Optimizer to be used."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Folder to store all checkpoints in. This folder will be created automatically if it does not exist.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, help="Checkpoint to start training from or a checkpoint to evaluate."
    )
    parser.add_argument(
        "--training_num_batches_per_epoch",
        type=int,
        default=512,
        help="Number of batches in a single epoch of training.",
    )
    parser.add_argument(
        "--backtest_id", type=int, default=-1, help="Backtest set to use. Use -1 to use the hyperparameter set."
    )
    parser.add_argument(
        "--prebacktest",
        action="store_true",
        help="When specified, uses the last few windows of the training set as the validation set. To be used only when training during backtesting. Not to be used when we are evaluating the model.",
    )
    parser.add_argument(
        "--log_subparams_every",
        type=int,
        default=10000,
        help="Frequency of logging the epoch number and iteration number during training.",
    )
    #parser.add_argument("--bagging_size", type=int, default=20, help="Bagging Size")
    parser.add_argument("--bagging_size", type=int, default=1, help="Bagging Size")

    parser.add_argument(
        "--dataset",
        type=str,
        default="fred_md",
        choices=[
            "fred_md",
            "kdd_cup_2018_without_missing",
            "solar_10min",
            "electricity_hourly",
            "traffic",
            "my_csv",

        ],
        help="Dataset to train on",
    )

    # Early stopping epochs based on total validation loss. -1 indicates no early stopping.
    parser.add_argument("--early_stopping_epochs", type=int, default=50, help="Early stopping patience")

    # HPARAMS
    # General ones
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight Decay")
    parser.add_argument("--clip_gradient", type=float, default=1e3, help="Gradient Clipping Magnitude")
    parser.add_argument("--history_factor", type=int, default=1, help="History Factor")

    parser.add_argument(
        "--validation_series_limit",
        type=int,
        default=None,
        help="Limit the number of series used for validation."
    )

    # Series embedding
    parser.add_argument(
        "--flow_series_embedding_dim", type=int, default=1, help="Embedding Dimension of the Flow Series Encoder"
    )
    parser.add_argument(
        "--copula_series_embedding_dim", type=int, default=1, help="Embedding Dimension of the Copula Series Encoder"
    )
    # Input embedding
    parser.add_argument(
        "--flow_input_encoder_layers", type=int, default=2, help="Embedding Dimension of the Flow Encoder"
    )
    parser.add_argument(
        "--copula_input_encoder_layers", type=int, default=2, help="Embedding Dimension of the Copula Encoder"
    )
    # Shared encoder
    parser.add_argument("--flow_encoder_num_layers", type=int, default=2, help="Number of Layers in the Flow Encoder")
    parser.add_argument("--flow_encoder_num_heads", type=int, default=1, help="Number of Heads in the Flow Encoder")
    parser.add_argument("--flow_encoder_dim", type=int, default=16, help="Embedding Dimension of the Flow Encoder")
    # Shared encoder
    parser.add_argument(
        "--copula_encoder_num_layers", type=int, default=2, help="Number of Layers in the Copula Encoder"
    )
    parser.add_argument("--copula_encoder_num_heads", type=int, default=1, help="Number of Heads in the Copula Encoder")
    parser.add_argument("--copula_encoder_dim", type=int, default=16, help="Embedding Dimension of the Copula Encoder")
    # Attentional Copula Decoder
    parser.add_argument("--decoder_num_layers", type=int, default=1, help="Number of Layers in the Attentional Copula")
    parser.add_argument("--decoder_num_heads", type=int, default=3, help="Number of Heads in the Attentional Copula")
    parser.add_argument("--decoder_dim", type=int, default=8, help="Embedding Dimension of the Attentional Copula")
    parser.add_argument(
        "--decoder_attention_mlp_class",
        type=str,
        default="_simple_linear_projection",
        choices=["_easy_mlp", "_simple_linear_projection"],
        help="MLP Type to be used in the Attentional Copula",
    )
    # Final layers in the decoder
    parser.add_argument("--decoder_resolution", type=int, default=20, help="Number of bins in the Attentional Copula")
    parser.add_argument(
        "--decoder_mlp_layers", type=int, default=2, help="Number of layers in the final MLP in the Decoder"
    )
    parser.add_argument(
        "--decoder_mlp_dim", type=int, default=48, help="Embedding Dimension of the final MLP in the Decoder"
    )
    parser.add_argument(
        "--decoder_act",
        type=str,
        default="relu",
        choices=["relu", "elu", "glu", "gelu"],
        help="Activation Function to be used in the Decoder",
    )
    # DSF Marginal
    parser.add_argument("--dsf_num_layers", type=int, default=2, help="Number of layers in the deep sigmoidal flow")
    parser.add_argument("--dsf_dim", type=int, default=48, help="Embedding Dimension of the deep sigmoidal flow")
    parser.add_argument(
        "--dsf_mlp_layers", type=int, default=2, help="Number of layers in the marginal conditioner MLP"
    )
    parser.add_argument(
        "--dsf_mlp_dim", type=int, default=48, help="Embedding Dimension of the marginal conditioner MLP"
    )

    # Loss normalization
    parser.add_argument(
        "--loss_normalization",
        type=str,
        default="both",
        choices=["", "none", "series", "timesteps", "both"],
        help="Loss normalization type",
    )

    # Modify this argument to use interpolation
    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=["forecasting", "interpolation"],
        default="forecasting",
        help="Operation mode of the model",
    )

    # Don't restrict memory / time
    parser.add_argument(
        "--do_not_restrict_memory",
        action="store_true",
        help="When enabled, memory is NOT restricted to 12 GB. Note that for all models in the paper, we used a GPU memory of 12 GB.",
    )
    parser.add_argument(
        "--do_not_restrict_time",
        action="store_true",
        help="When enabled, total training time is NOT restricted to 3 days. Note that for all models in the paper, we used a maximum training time of 3 days.",
    )

    # Skip batch size search
    parser.add_argument(
        "--skip_batch_size_search",
        action="store_true",
        help="When enabled, batch size search is NOT performed. Note that for all models in the paper, we used the batch size search to maximize the batch size within the 12 GB GPU memory constraint.",
    )

    # CPU
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="When enabled, CPU is used instead of GPU"
    )

    # Flag for evaluation (either NLL or sampling and metrics)
    # A checkpoint must be provided for evaluation
    # Note evaluation is only supported after training the model in both phases.
    parser.add_argument("--evaluate", action="store_true", help="Evaluate for NLL and metrics.")

    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    main(args=args)