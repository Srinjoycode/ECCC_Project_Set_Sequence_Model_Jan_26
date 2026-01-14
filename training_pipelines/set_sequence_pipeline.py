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
            # If the basin is too short for even one sequence, try another basin
            return self.__getitem__(np.random.randint(0, self.n_basins))

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

    # Determine filtering flags based on FILTERING_MODE
    mode = config.FILTERING_MODE
    filter_pretrain = (mode == 'all')
    filter_finetune = (mode in ['finetune', 'all'])
    filter_eval = (mode in ['eval_only', 'finetune', 'all'])

    print(f"Filtering Mode: {mode}")
    print(f"  - Filter Pretrain Data: {filter_pretrain}")
    print(f"  - Filter Finetune Data: {filter_finetune}")
    print(f"  - Filter Evaluation:    {filter_eval}")

    # ====================================================
    # STEP 1: LOAD PRETRAINING DATA (3D STACKED)
    # ====================================================
    # Note: flatten_spatial=False gives (Time, Grid, Feat)
    #X_list, M_list, y_list, successful_basins, months_list
    X_list, pretrain_M_list, y_list, successful_basins, pretrain_months_list = load_all_pretrain_basins(
        basin_list, config.CAMELS_SPAT_ROOT,
        apply_seasonal_filter=filter_pretrain,
        target_months=config.SEASONAL_MONTHS,
        config=config,
        flatten_spatial=False
    )

    print(f"#Dataset of pretraining loaded from {len(successful_basins)}")
    print(f"X_list shape: {len(X_list)}")
    print(f"y_list shape: {len(y_list)}")
    print(f"M_list shape: {len(pretrain_M_list)}")
    print(f"months_list shape: {len(pretrain_months_list)}")


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
    # 1. Feature Normalization (X)
    # Applied globally (across all basins) to preserve relative climatology signals (e.g. Arid vs Wet basins)
    print("Fitting Global Feature Scaler (X)...")
    x_scaler = fit_feature_scaler_from_basin_list(X_train_list)

    X_train_list = [x_scaler.transform(Xb).astype(DEFAULT_DTYPE) for Xb in X_train_list]
    X_val_list = [x_scaler.transform(Xb).astype(DEFAULT_DTYPE) for Xb in X_val_list]
    
    # 2. Target Normalization (y) - PER BASIN
    # CRITICAL CHANGE: We normalize flow *per basin* to N(0,1).
    # This prevents large rivers from dominating the gradient and allows the model 
    # to learn the *shape* of the hydrograph response function universally.
    print("Applying Per-Basin Target Scaling (y)...")
    
    y_train_list_norm = []
    y_val_list_norm = []
    
    # We don't keep the scalers for pretraining since we don't need to inverse predict 
    # for specific pretraining basins (we only care about the weights for the ROI later).
    for i, (yb_train, yb_val) in enumerate(zip(y_train_list, y_val_list)):
        # Fit on training portion only
        local_scaler = StandardScalerNP().fit(yb_train.reshape(-1, 1))
        
        # Transform both train and val
        y_train_norm = local_scaler.transform(yb_train).astype(DEFAULT_DTYPE)
        y_val_norm = local_scaler.transform(yb_val).astype(DEFAULT_DTYPE)
        
        y_train_list_norm.append(y_train_norm)
        y_val_list_norm.append(y_val_norm)
        
    # Replace the lists with normalized versions
    y_train_list = y_train_list_norm
    y_val_list = y_val_list_norm

    # --- DYNAMIC STEPS CALCULATION ---
    total_train_sequences = sum([max(0, len(X) - config.SEQ_LENGTH - config.PREDICT_AHEAD) for X in X_train_list])
    total_val_sequences = sum([max(0, len(X) - config.SEQ_LENGTH - config.PREDICT_AHEAD) for X in X_val_list])
    
    # Heuristic: Cover the dataset roughly once per epoch
    # Ensure at least 1 step
    steps_per_epoch = max(1, total_train_sequences // config.BATCH_SIZE)
    val_steps = max(1, total_val_sequences // config.BATCH_SIZE)

    print("\n--- Pretraining Data Stats ---")
    print(f"Total Train Sequences Available: {total_train_sequences}")
    print(f"Total Val Sequences Available:   {total_val_sequences}")
    print(f"Calculated Steps Per Epoch:      {steps_per_epoch} (Batch Size: {config.BATCH_SIZE})")
    print(f"Calculated Val Steps:            {val_steps}")

    for i, (Xb, yb) in enumerate(zip(X_train_list, y_train_list)):
         print(f"Basin {i}: X shape={Xb.shape}, y shape={yb.shape}")
         print("temporal length:", Xb.shape[0])
    
    # Create Generators
    # WRAPPER FIX: We must wrap the generator in a tf.data.Dataset to allow dynamic shapes.
    # Keras Sequence enforces static shapes based on the first batch, which crashes with variable grid sizes.
    
    def generator_wrapper(gen):
        for i in range(len(gen)):
            yield gen[i]

    # Define output signature with None for grid dimension
    output_signature = (
        tf.TensorSpec(shape=(None, config.SEQ_LENGTH, None, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    train_gen_base = BasinSequenceGenerator(
        X_train_list, y_train_list,
        batch_size=config.BATCH_SIZE,
        seq_length=config.SEQ_LENGTH,
        steps_per_epoch=steps_per_epoch,
        predict_ahead=getattr(config, 'PREDICT_AHEAD', 1)
    )
    
    val_gen_base = BasinSequenceGenerator(
        X_val_list, y_val_list,
        batch_size=config.BATCH_SIZE,
        seq_length=config.SEQ_LENGTH,
        steps_per_epoch=val_steps,
        predict_ahead=getattr(config, 'PREDICT_AHEAD', 1)
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator_wrapper(train_gen_base),
        output_signature=output_signature
    ).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generator_wrapper(val_gen_base),
        output_signature=output_signature
    ).repeat().prefetch(tf.data.AUTOTUNE)

    print(f"\nBuilding Set-Sequence Model...")
    model = build_set_sequence_model(
        seq_length=config.SEQ_LENGTH,
        n_features=n_features,
        latent_dim=config.LATENT_DIM,
        lstm_units=config.LSTM_UNITS,
        lstm_layers=config.LSTM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    )

    # Use MSE loss for pretraining on locally normalized targets.
    # Since targets are ~N(0,1), MSE is stable and effectively optimizes correlation/Nash-Sutcliffe.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_PRETRAIN),
        loss='mse', 
        metrics=['mae', 'mse']
    )
    model.summary()

    print("\n--- Starting Pretraining ---")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_PRETRAIN,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=getattr(config, 'PATIENCE_PRETRAIN', 10), restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=getattr(config, 'LR_FACTOR', 0.5), patience=getattr(config, 'LR_PATIENCE', 5)),
            ModelCheckpoint(os.path.join(output_dir, 'best_set_seq_pretrain.keras'), save_best_only=True)
        ]
    )
    plot_training_history(history, "Set_Seq_Pretrain", output_dir)

    # Clean up pretraining data
    del X_list, y_list, X_train_list, X_val_list, train_gen_base, val_gen_base, train_dataset, val_dataset
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
        apply_seasonal_filter=filter_finetune,
        target_months=config.SEASONAL_MONTHS,
        config=config,
        flatten_spatial=False
    )

    # Create Sequences for ROI
    # For ROI, we have a single dataset, so we can create standard arrays
    # X_roi is (Time, 1, Feat)


    # MODIFIED: Now returns months sequence as well
    X_roi_seq, y_roi_seq, roi_months_seq = create_sequences(
        X_roi, X_global, y_roi,
        config.SEQ_LENGTH,
        config.PREDICT_AHEAD,
        months=roi_months
    )

    # Train/Val/Test Split for ROI
    train_size = int(len(X_roi_seq) * (1 - config.TEST_SPLIT_FRACTION))
    X_train_roi = X_roi_seq[:train_size]
    y_train_roi = y_roi_seq[:train_size]
    X_test_roi = X_roi_seq[train_size:]
    y_test_roi = y_roi_seq[train_size:]
    
    # Also split months for evaluation filtering
    months_test = roi_months_seq[train_size:]

    # ------------------------------
    # NORMALIZATION (ROI)
    # ------------------------------
    # CONSISTENCY FIX: Use the SAME x_scaler from pretraining for the ROI inputs.
    # This ensures the model sees features in the same global distribution space.
    # We do NOT refit X scaler on the ROI.
    print("Applying Pretraining Global Scaler to ROI inputs...")

    # Check if x_scaler exists (it should from the pretraining block above)
    if 'x_scaler' not in locals():
        print("WARNING: x_scaler not found. Fitting local scaler (suboptimal for consistency).")
        x_roi_scaler = StandardScalerNP().fit(X_train_roi.reshape(-1, X_train_roi.shape[-1]))
    else:
        x_roi_scaler = x_scaler

    # Target (y) is still locally normalized (Per-Basin / Per-ROI)
    # This matches the 'Per-Basin Target Scaling' used in pretraining.
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
        loss='mse',
        metrics=['mae']
    )

    print("\n--- Starting ROI Fine-tuning ---")
    print("ROI Train")
    print(f"X_train_roi_n shape: {X_train_roi_n.shape}")
    print(f"y_train_roi_n shape:{y_train_roi_n.shape}")
    print("ROI Test")
    print(f"X_test_roi_n shape: {X_test_roi_n.shape}")
    print(f"y_test_roi_n shape: {y_test_roi_n.shape}")
    
    print(f"Fine-tuning: {len(X_train_roi_n)} training samples")
    print(f"Fine-tuning: {len(X_test_roi_n)} validation/test samples")

    history_ft = model.fit(
        X_train_roi_n, y_train_roi_n,
        validation_data=(X_test_roi_n, y_test_roi_n),
        epochs=config.EPOCHS_FINETUNE,
        batch_size=config.BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=getattr(config, 'PATIENCE_FINETUNE', 15), restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=getattr(config, 'LR_FACTOR', 0.5), patience=getattr(config, 'LR_PATIENCE', 5)),
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
    
    # APPLY EVALUATION FILTERING IF REQUESTED
    if filter_eval and config.SEASONAL_MONTHS:
        print(f"Filtering evaluation to seasonal months: {config.SEASONAL_MONTHS}")
        # Create mask based on months_test
        eval_mask = np.isin(months_test, config.SEASONAL_MONTHS)
        
        y_test_eval = y_test[eval_mask]
        y_pred_eval = y_pred[eval_mask]
        
        print(f"Filtered evaluation set size: {len(y_test_eval)} / {len(y_test)}")
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred

    metrics = compute_basin_metrics(y_test_eval, y_pred_eval)
    print(f"Test Metrics: {metrics}")

    plot_predictions(
        y_test_eval, y_pred_eval,
        "Set_Seq_Predictions",
        output_dir,
        filter_months=config.SEASONAL_MONTHS if filter_eval else None,
        target_months_seq=None
    )

    return {
        'pipeline': 'Set-Sequence',
        'output_dir': output_dir,
        'metrics': metrics
    }