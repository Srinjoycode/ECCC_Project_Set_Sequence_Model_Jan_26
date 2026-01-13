import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils.memory_utils import print_memory_status, clear_memory
from utils.set_seq_data_loading import load_all_pretrain_basins, process_roi_csv
from utils.set_sequence_data_processing import create_sequences
from models.set_sequence import build_set_sequence_model
from visualisation import plot_training_history, plot_predictions
from utils.metrics_utils import compute_basin_metrics
from utils.normalization_utils import StandardScalerNP, fit_feature_scaler_from_basin_list

DEFAULT_DTYPE = np.float32


class NSELoss(tf.keras.losses.Loss):
    """
    Nash-Sutcliffe Efficiency (NSE) Loss.
    Target: Maximize NSE -> Minimize (1 - NSE).

    Formula:
    Loss = Sum((y_true - y_pred)^2) / (Sum((y_true - mean(y_true))^2) + epsilon)

    This is effectively MSE normalized by the variance of the target.
    It allows the model to learn equally from small and large basins.
    """

    def __init__(self, name="nse_loss", epsilon=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        # Flatten to ensure shapes match (Batch, 1) -> (Batch,)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # 1. Compute Mean Squared Error (Numerator)
        numerator = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Compute Variance of Target (Denominator)
        # We use the variance within the batch as a proxy for basin variance
        y_mean = tf.reduce_mean(y_true)
        denominator = tf.reduce_mean(tf.square(y_true - y_mean))

        # 3. Compute NSE Loss
        # If variance is 0 (constant flow), add epsilon to prevent division by zero
        loss = numerator / (denominator + self.epsilon)

        return loss


class BasinSequenceGenerator(tf.keras.utils.Sequence):
    """Custom generator; within a batch, grid size is constant (per basin)."""

    def __init__(self, X_list, y_list, batch_size, seq_length, steps_per_epoch=100, predict_ahead=1):
        super().__init__()
        self.X_list = X_list
        self.y_list = y_list
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.steps_per_epoch = steps_per_epoch
        self.predict_ahead = int(predict_ahead)
        self.n_basins = len(X_list)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        basin_idx = np.random.randint(0, self.n_basins)
        X_basin = self.X_list[basin_idx]
        y_basin = self.y_list[basin_idx]

        # Need enough room for seq_length + predict_ahead target
        max_start = len(X_basin) - (self.seq_length + self.predict_ahead)
        if max_start <= 0:
            return self.__getitem__(index)

        start_indices = np.random.randint(0, max_start + 1, self.batch_size)

        batch_X, batch_y = [], []
        for start in start_indices:
            end = start + self.seq_length
            target_idx = end + self.predict_ahead - 1
            batch_X.append(X_basin[start:end])
            batch_y.append(y_basin[target_idx])

        return np.array(batch_X), np.array(batch_y)


def pipeline_set_sequence(config, basin_list, args):
    print(f"\n{'#' * 60}")
    print("PIPELINE: Set-Sequence Model (Unified Pretrain + Finetune)")
    print(f"{'#' * 60}\n")
    print_memory_status("Set-Sequence Pipeline Start")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config.PRETRAIN_OUTPUT_DIR, f"set_seq_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # ====================================================
    # STEP 1: LOAD PRETRAINING DATA (3D STACKED)
    # ====================================================
    # Note: flatten_spatial=False gives (Time, Grid, Feat)
    X_list, _, y_list, successful_basins, _ = load_all_pretrain_basins(
        basin_list, config.CAMELS_SPAT_ROOT,
        apply_seasonal_filter=False,
        config=config,
        flatten_spatial=False
    )

    # ====================================================
    # STEP 2: BUILD & PRETRAIN SET-SEQUENCE MODEL
    # ====================================================
    # Determine input dimensions from first basin
    sample_basin = X_list[0]
    n_features = sample_basin.shape[-1]  # Number of meteo variables
    print(f"Features per grid point: {n_features}")

    # Split basins into Train/Val sets (Basin-wise split)
    # Alternatively, you can split temporally per basin. Let's do temporal per basin for robustness.
    X_train_list, X_val_list = [], []
    y_train_list, y_val_list = [], []

    for X, y in zip(X_list, y_list):
        split_idx = int(len(X) * (1 - config.TEST_SPLIT_FRACTION))
        X_train_list.append(X[:split_idx])
        y_train_list.append(y[:split_idx])
        X_val_list.append(X[split_idx:])
        y_val_list.append(y[split_idx:])

    # ------------------------------
    # NORMALIZATION (PRETRAIN)
    # ------------------------------
    # Fit scalers on TRAIN only; apply to train/val.
    # X scaler is per-feature computed across all train basins/time/grid.
    x_scaler = fit_feature_scaler_from_basin_list(X_train_list)

    # y scaler is a single-dim standard scaler fit across all train basins.
    y_scaler = StandardScalerNP().fit(np.concatenate([yb.reshape(-1, 1) for yb in y_train_list], axis=0))

    X_train_list = [x_scaler.transform(Xb).astype(DEFAULT_DTYPE) for Xb in X_train_list]
    X_val_list = [x_scaler.transform(Xb).astype(DEFAULT_DTYPE) for Xb in X_val_list]
    y_train_list = [y_scaler.transform(yb).astype(DEFAULT_DTYPE) for yb in y_train_list]
    y_val_list = [y_scaler.transform(yb).astype(DEFAULT_DTYPE) for yb in y_val_list]

    # Create Generators
    train_gen = BasinSequenceGenerator(
        X_train_list, y_train_list,
        batch_size=config.BATCH_SIZE,
        seq_length=config.SEQ_LENGTH,
        steps_per_epoch=getattr(config, 'STEPS_PER_EPOCH', 100),
        predict_ahead=getattr(config, 'PREDICT_AHEAD', 1)
    )
    val_gen = BasinSequenceGenerator(
        X_val_list, y_val_list,
        batch_size=config.BATCH_SIZE,
        seq_length=config.SEQ_LENGTH,
        steps_per_epoch=getattr(config, 'VAL_STEPS', 20),
        predict_ahead=getattr(config, 'PREDICT_AHEAD', 1)
    )

    print(f"\nBuilding Set-Sequence Model...")
    model = build_set_sequence_model(
        seq_length=config.SEQ_LENGTH,
        n_features=n_features,
        latent_dim=config.LATENT_DIM,
        lstm_units=config.LSTM_UNITS,
        lstm_layers=config.LSTM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_PRETRAIN),
        #loss=NSELoss(),
        loss='msle',
        metrics=['mae']
    )
    model.summary()

    print("\n--- Starting Pretraining ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_PRETRAIN,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            ModelCheckpoint(os.path.join(output_dir, 'best_set_seq_pretrain.keras'), save_best_only=True)
        ]
    )
    plot_training_history(history, "Set_Seq_Pretrain", output_dir)

    # Clean up pretraining data
    del X_list, y_list, X_train_list, X_val_list, train_gen, val_gen
    clear_memory(verbose=True)

    # ====================================================
    # STEP 3: FINE-TUNE ON ROI (Region of Interest)
    # ====================================================
    print(f"\n{'=' * 60}")
    print("FINE-TUNING ON ROI")
    print(f"{'=' * 60}")

    # Load ROI data (flatten_spatial=False -> Grid=1)
    X_roi, X_global, y_roi, roi_months = process_roi_csv(
        config.ROI_DATA_PATH,
        config=config,
        flatten_spatial=False
    )

    # Create Sequences for ROI
    # For ROI, we have a single dataset, so we can create standard arrays
    # X_roi is (Time, 1, Feat)


    X_roi_seq, y_roi_seq = create_sequences(
        X_roi, X_global, y_roi,
        config.SEQ_LENGTH,
        config.PREDICT_AHEAD
    )#x_global is not real for now

    # Train/Val/Test Split for ROI
    train_size = int(len(X_roi_seq) * (1 - config.TEST_SPLIT_FRACTION))
    X_train_roi = X_roi_seq[:train_size]
    y_train_roi = y_roi_seq[:train_size]
    X_test_roi = X_roi_seq[train_size:]
    y_test_roi = y_roi_seq[train_size:]

    # ------------------------------
    # NORMALIZATION (ROI)
    # ------------------------------
    # For ROI we fit scalers on ROI-train only. This keeps evaluation honest.
    x_roi_scaler = StandardScalerNP().fit(X_train_roi.reshape(-1, X_train_roi.shape[-1]))
    y_roi_scaler = StandardScalerNP().fit(y_train_roi.reshape(-1, 1))

    X_train_roi_n = x_roi_scaler.transform(X_train_roi).astype(DEFAULT_DTYPE)
    X_test_roi_n = x_roi_scaler.transform(X_test_roi).astype(DEFAULT_DTYPE)
    y_train_roi_n = y_roi_scaler.transform(y_train_roi).astype(DEFAULT_DTYPE)
    y_test_roi_n = y_roi_scaler.transform(y_test_roi).astype(DEFAULT_DTYPE)

    print(f"ROI Train shape: {X_train_roi.shape}")
    print(f"ROI Test shape: {X_test_roi.shape}")

    # Compile for fine-tuning (lower LR)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_FINETUNE),
        #loss=NSELoss(),
        loss='msle',
        metrics=['mae']
    )

    print("\n--- Starting ROI Fine-tuning ---")
    print("ROI Train")
    print(f"X_train_roi_n shape: {X_train_roi_n.shape}")
    print(f"y_train_roi_n shape:{y_train_roi_n.shape}")
    print("ROI Test")
    print(f"X_test_roi_n shape: {X_test_roi_n.shape}")
    print(f"y_test_roi_n shape: {y_test_roi_n.shape}")

    history_ft = model.fit(
        X_train_roi_n, y_train_roi_n,
        validation_data=(X_test_roi_n, y_test_roi_n),
        epochs=config.EPOCHS_FINETUNE,
        batch_size=config.BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(os.path.join(output_dir, 'best_set_seq_finetune.keras'), save_best_only=True)
        ]
    )
    plot_training_history(history_ft, "Set_Seq_Finetune", output_dir)

    # ====================================================
    # STEP 4: EVALUATION
    # ====================================================
    print("\n--- Evaluation ---")

    # Predict in normalized space then inverse-transform for metrics/plots
    y_pred_n = model.predict(X_test_roi_n)
    y_pred = y_roi_scaler.inverse_transform(y_pred_n)
    y_test = y_test_roi  # already raw

    metrics = compute_basin_metrics(y_test, y_pred)
    print(f"Test Metrics: {metrics}")

    plot_predictions(
        y_test, y_pred,
        "Set_Seq_Predictions",
        output_dir,
        filter_months=None,
        target_months_seq=None
    )

    return {
        'pipeline': 'Set-Sequence',
        'output_dir': output_dir,
        'metrics': metrics
    }