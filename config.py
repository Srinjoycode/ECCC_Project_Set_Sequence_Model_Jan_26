# ==========================================
# UNIFIED CONFIGURATION SYSTEM
# ==========================================
import os,json

class Config:
    """
    Unified Configuration class for the training pipeline.
    """

    def __init__(self, config_file=None):
        # ==========================================
        # PATHS
        # ==========================================
        self.CAMELS_SPAT_ROOT = "/media/sbhuiya/1a899d3a-b2a4-487c-b59c-fd2cac4442c8/CAMELS-SPAT"
        self.BASIN_LIST_PATH = "basin_lists/camels_pretraining_basin_list.txt"
        self.ROI_DATA_PATH = "FinalData/Dataset4.csv"

        # Single-root output (all artifacts should derive from this)
        # If OUTPUT_ROOT_DIR is set, PRETRAIN_OUTPUT_DIR/FINETUNE_OUTPUT_DIR/LATENTS_SAVE_DIR are derived from it.
        self.OUTPUT_ROOT_DIR = "Results/large1_V2_MODEL_mse_pre_fine_shared_ae_5multi-horizon_plots_fixed_and_filter_sequences_skipmode"

        # Paths for saving/loading pre-computed latent representations
        self.SAVE_LATENTS = True
        self.LOAD_LATENTS = False
        self.LATENTS_SAVE_DIR = None  # Derived from OUTPUT_ROOT_DIR if None
        self.LATENTS_LOAD_PATH = None

        # ==========================================
        # MEMORY / PERFORMANCE
        # ==========================================
        # Number of basins to process before triggering garbage collection / chunk boundary
        self.BASIN_PROCESSING_BATCH_SIZE = 50

        # ==========================================
        # MODEL ARCHITECTURE
        # ==========================================
        self.LATENT_DIM = 512
        self.LSTM_UNITS = 512
        self.LSTM_LAYERS = 2
        self.SEQ_LENGTH = 12
        self.PREDICT_AHEAD = 1

        # Multistep forecasting
        # FORECAST_HORIZON = Number of future months predicted per input sequence.
        # - 1 keeps legacy 1-step behavior.
        # - H>1 enables multistep vector outputs.
        self.FORECAST_HORIZON = 5

        # Evaluation aggregation for multistep outputs.
        # 'none': evaluate each horizon step separately (and optionally flattened)
        # 'mean': aggregate overlapping predictions per target month by mean (running average)
        self.EVAL_MULTI_STEP_AGGREGATION = 'mean'

        self.DROPOUT_RATE = 0.4
        self.GRU_UNITS = 512
        self.NUM_ATTENTION_HEADS = 8
        self.USE_SHARED_AUTOENCODER = True
        self.FLOW_HISTORY_LAG_GAP = 0

        # ==========================================
        # MODEL VERSION CONTROL (V1 vs V2)
        # ==========================================
        self.USE_MODEL_V2 = True  # If True, use V2 with joint same joint input in 2 branches; if False, use V1 dual-branch

        # ==========================================
        # TRAINING HYPERPARAMETERS
        # ==========================================
        self.BATCH_SIZE = 1024
        self.EPOCHS_AE = 100
        self.EPOCHS_PRETRAIN = 300
        self.EPOCHS_FINETUNE = 250
        self.LEARNING_RATE_AE = 0.0005
        self.LEARNING_RATE_PRETRAIN = 0.0005
        self.LEARNING_RATE_FINETUNE = 0.0005
        self.TEST_SPLIT_FRACTION = 0.23

        # ==========================================
        # CALLBACK / TRAINING CONTROL (NO HARDCODE)
        # ==========================================
        # Early stopping
        self.EARLY_STOPPING_MONITOR = 'val_loss'
        self.EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
        self.EARLY_STOPPING_PATIENCE_AE = 30
        self.EARLY_STOPPING_PATIENCE_PRETRAIN = 30
        self.EARLY_STOPPING_PATIENCE_FINETUNE = 30

        # ReduceLROnPlateau
        self.REDUCE_LR_MONITOR = 'val_loss'
        self.REDUCE_LR_FACTOR = 0.5
        self.REDUCE_LR_PATIENCE_AE = 10
        self.REDUCE_LR_PATIENCE_PRETRAIN = 15
        self.REDUCE_LR_PATIENCE_FINETUNE = 10
        self.REDUCE_LR_MIN_LR_AE = 1e-7
        self.REDUCE_LR_MIN_LR_PRETRAIN = 1e-7
        self.REDUCE_LR_MIN_LR_FINETUNE = 1e-8

        # Gradient clipping
        self.GRADIENT_CLIPNORM = 1.0

        # ==========================================
        # FEATURE ENGINEERING
        # ==========================================
        self.SEASONAL_MONTHS = [5, 6, 7, 8, 9]
        self.MONTH_ENCODING = 'sinusoidal'
        self.MONTH_EMB_DIM = 12
        self.INCLUDE_FLOW_HISTORY = True

        # ==========================================
        # LOSS FUNCTION CONFIGURATION
        # ==========================================
        self.LOSS_FUNCTION_AE = 'mse'
        self.LOSS_FUNCTION_PRETRAIN = 'mse'
        self.LOSS_FUNCTION_FINETUNE = 'mse'
        self.NSE_EPSILON = 1e-6
        self.COMBINED_LOSS_ALPHA = 0.5
        self.LOSS_FUNCTION = None

        # ==========================================
        # OUTPUT DIRECTORIES (DERIVED)
        # ==========================================
        self.PRETRAIN_OUTPUT_DIR = None
        self.FINETUNE_OUTPUT_DIR = None

        # ==========================================
        # MULTI-STAGE FINE-TUNING SCHEDULE
        # ==========================================
        self.MULTISTAGE_SCHEDULE = {
            1: {"epochs": 150, "lr_factor": 1.0},
            2: {"epochs": 155, "lr_factor": 0.8},
            3: {"epochs": 170, "lr_factor": 0.5},
            4: {"epochs": 200, "lr_factor": 0.5},
            5: {"epochs": 350, "lr_factor": 0.3},
        }

        # ==========================================
        # AUTOENCODER TRAINING CONTROL
        # ==========================================
        self.AE_EPOCHS_PER_BASIN = 100
        self.AE_PATIENCE = 30

        # ==========================================
        # PIPELINE CONTROL
        # ==========================================
        self.PIPELINE_VERSION = 'all'
        self.EVAL_FILTERED = True
        self.SEED = 42

        # ==========================================
        # AUTOENCODER RECONSTRUCTION EVALUATION
        # ==========================================
        # Evaluate feature reconstruction quality on the *existing AE validation split*.
        self.EVAL_AE_RECONSTRUCTION = True

        # Which feature space to compute reconstruction metrics in:
        # - 'scaled': on StandardScaler-transformed features (always available)
        # - 'original': inverse-transformed back to the original feature scale(unscaled)
        # - 'both': compute both
        self.AE_RECON_EVAL_SPACE = 'both'

        # Save validation reconstructions (y_true/y_pred) for plotting/debugging
        self.SAVE_AE_RECONSTRUCTIONS = True

        # Cap the number of validation samples saved per basin (None = save all)
        self.AE_RECON_MAX_SAMPLES = None

        # Metrics output file name (written in output_dir; JSONL format)
        self.AE_RECON_METRICS_FILENAME = 'ae_reconstruction_metrics.jsonl'

        # If True, save per-basin AE models to disk (can be large); metrics logging works without this.
        self.SAVE_PER_BASIN_AE_MODELS = False

        # ==========================================
        # SEASONAL / SEQUENCE TIMELINE MODE
        # ==========================================
        # SEASONAL_SEQUENCE_MODE controls how sequences interpret time steps:
        # - 'contiguous' (default): existing behavior on the full monthly timeline
        # - 'skip'                : build sequences on an allowed-month timeline (next step is next allowed month)
        #
        # When 'skip' is used, seq_length/predict_ahead/forecast_horizon are measured in allowed-month steps.
        self.SEASONAL_SEQUENCE_MODE = 'skip'

        # Set which months define the allowed-month timeline for skip mode,if none, defaults to SEASONAL_MONTHS.
        self.SEASONAL_SEQUENCE_ALLOWED_MONTHS = [5, 6, 7, 8, 9]

        # Apply skip-mode to which stages:
        # - 'none' (default)
        # - 'pretrain'
        # - 'finetune'
        # - 'pretrain+finetune'
        self.SEASONAL_SEQUENCE_APPLY_TO = 'pretrain+finetune'

        if config_file is not None:
            self.load_from_file(config_file)

        # Always finalize derived outputs after loading overrides
        self._finalize_paths()

    def _finalize_paths(self):
        """Derive all output paths from OUTPUT_ROOT_DIR unless explicitly provided."""
        root = self.OUTPUT_ROOT_DIR
        if self.PRETRAIN_OUTPUT_DIR is None:
            self.PRETRAIN_OUTPUT_DIR = os.path.join(root, 'pretraining')
        if self.FINETUNE_OUTPUT_DIR is None:
            self.FINETUNE_OUTPUT_DIR = os.path.join(root, 'finetuning')
        if self.LATENTS_SAVE_DIR is None:
            self.LATENTS_SAVE_DIR = os.path.join(root, 'latents')

    def load_from_file(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            attr_name = key.upper()
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)
            else:
                print(f"Warning: Unknown config parameter '{key}' in file, ignoring.")

    def save_to_file(self, filepath):
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Configuration saved to: {filepath}")

    def to_dict(self):
        return {
            # Paths
            'camels_spat_root': self.CAMELS_SPAT_ROOT,
            'basin_list_path': self.BASIN_LIST_PATH,
            'roi_data_path': self.ROI_DATA_PATH,
            'output_root_dir': self.OUTPUT_ROOT_DIR,
            'pretrain_output_dir': self.PRETRAIN_OUTPUT_DIR,
            'finetune_output_dir': self.FINETUNE_OUTPUT_DIR,
            'save_latents': self.SAVE_LATENTS,
            'load_latents': self.LOAD_LATENTS,
            'latents_save_dir': self.LATENTS_SAVE_DIR,
            'latents_load_path': self.LATENTS_LOAD_PATH,
            # Memory
            'basin_processing_batch_size': self.BASIN_PROCESSING_BATCH_SIZE,
            # Model Architecture
            'latent_dim': self.LATENT_DIM,
            'lstm_units': self.LSTM_UNITS,
            'lstm_layers': self.LSTM_LAYERS,
            'seq_length': self.SEQ_LENGTH,
            'predict_ahead': self.PREDICT_AHEAD,
            'forecast_horizon': self.FORECAST_HORIZON,
            'eval_multi_step_aggregation': self.EVAL_MULTI_STEP_AGGREGATION,
            'dropout_rate': self.DROPOUT_RATE,
            'gru_units': self.GRU_UNITS,
            'num_attention_heads': self.NUM_ATTENTION_HEADS,
            'use_model_v2': self.USE_MODEL_V2,
            'use_shared_autoencoder': self.USE_SHARED_AUTOENCODER,
            'flow_history_lag_gap': self.FLOW_HISTORY_LAG_GAP,
            # Training Hyperparameters
            'batch_size': self.BATCH_SIZE,
            'epochs_ae': self.EPOCHS_AE,
            'epochs_pretrain': self.EPOCHS_PRETRAIN,
            'epochs_finetune': self.EPOCHS_FINETUNE,
            'learning_rate_ae': self.LEARNING_RATE_AE,
            'learning_rate_pretrain': self.LEARNING_RATE_PRETRAIN,
            'learning_rate_finetune': self.LEARNING_RATE_FINETUNE,
            'test_split_fraction': self.TEST_SPLIT_FRACTION,
            # Callback params
            'early_stopping_monitor': self.EARLY_STOPPING_MONITOR,
            'early_stopping_restore_best_weights': self.EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
            'early_stopping_patience_ae': self.EARLY_STOPPING_PATIENCE_AE,
            'early_stopping_patience_pretrain': self.EARLY_STOPPING_PATIENCE_PRETRAIN,
            'early_stopping_patience_finetune': self.EARLY_STOPPING_PATIENCE_FINETUNE,
            'reduce_lr_monitor': self.REDUCE_LR_MONITOR,
            'reduce_lr_factor': self.REDUCE_LR_FACTOR,
            'reduce_lr_patience_ae': self.REDUCE_LR_PATIENCE_AE,
            'reduce_lr_patience_pretrain': self.REDUCE_LR_PATIENCE_PRETRAIN,
            'reduce_lr_patience_finetune': self.REDUCE_LR_PATIENCE_FINETUNE,
            'reduce_lr_min_lr_ae': self.REDUCE_LR_MIN_LR_AE,
            'reduce_lr_min_lr_pretrain': self.REDUCE_LR_MIN_LR_PRETRAIN,
            'reduce_lr_min_lr_finetune': self.REDUCE_LR_MIN_LR_FINETUNE,
            'gradient_clipnorm': self.GRADIENT_CLIPNORM,
            # Feature Engineering
            'seasonal_months': self.SEASONAL_MONTHS,
            'month_encoding': self.MONTH_ENCODING,
            'month_emb_dim': self.MONTH_EMB_DIM,
            'include_flow_history': self.INCLUDE_FLOW_HISTORY,
            # Loss Functions
            'loss_function_ae': self.LOSS_FUNCTION_AE,
            'loss_function_pretrain': self.LOSS_FUNCTION_PRETRAIN,
            'loss_function_finetune': self.LOSS_FUNCTION_FINETUNE,
            'loss_function': self.LOSS_FUNCTION,
            'nse_epsilon': self.NSE_EPSILON,
            'combined_loss_alpha': self.COMBINED_LOSS_ALPHA,
            # Pipeline Control
            'pipeline_version': self.PIPELINE_VERSION,
            'eval_filtered': self.EVAL_FILTERED,
            'seed': self.SEED,
            # Multi-stage
            'multistage_schedule': self.MULTISTAGE_SCHEDULE,

            # AE reconstruction evaluation
            'eval_ae_reconstruction': self.EVAL_AE_RECONSTRUCTION,
            'ae_recon_eval_space': self.AE_RECON_EVAL_SPACE,
            'save_ae_reconstructions': self.SAVE_AE_RECONSTRUCTIONS,
            'ae_recon_max_samples': self.AE_RECON_MAX_SAMPLES,
            'ae_recon_metrics_filename': self.AE_RECON_METRICS_FILENAME,
            'save_per_basin_ae_models': self.SAVE_PER_BASIN_AE_MODELS,
        }

    @classmethod
    def from_args(cls, args):
        # Load from the file if provided
        if hasattr(args, 'config_file') and args.config_file:
            config = cls(config_file=args.config_file)
            print(f"✅ Loaded base config from: {args.config_file}")
        else:
            config = cls()

        # ---------- Standard overrides ----------
        if hasattr(args, 'basin_list') and args.basin_list is not None:
            config.BASIN_LIST_PATH = args.basin_list
        if hasattr(args, 'filter_months') and args.filter_months is not None:
            config.SEASONAL_MONTHS = args.filter_months
        if hasattr(args, 'latent_dim') and args.latent_dim is not None:
            config.LATENT_DIM = args.latent_dim
        if hasattr(args, 'lstm_units') and args.lstm_units is not None:
            config.LSTM_UNITS = args.lstm_units
        if hasattr(args, 'seq_length') and args.seq_length is not None:
            config.SEQ_LENGTH = args.seq_length
        if hasattr(args, 'month_encoding') and args.month_encoding is not None:
            config.MONTH_ENCODING = args.month_encoding
        if hasattr(args, 'month_emb_dim') and args.month_emb_dim is not None:
            config.MONTH_EMB_DIM = args.month_emb_dim
        if hasattr(args, 'include_flow_history') and args.include_flow_history is not None:
            config.INCLUDE_FLOW_HISTORY = args.include_flow_history

        if hasattr(args, 'epochs_ae') and args.epochs_ae is not None:
            config.EPOCHS_AE = args.epochs_ae
        if hasattr(args, 'epochs_pretrain') and args.epochs_pretrain is not None:
            config.EPOCHS_PRETRAIN = args.epochs_pretrain
        if hasattr(args, 'epochs_finetune') and args.epochs_finetune is not None:
            config.EPOCHS_FINETUNE = args.epochs_finetune

        if hasattr(args, 'lr_ae') and args.lr_ae is not None:
            config.LEARNING_RATE_AE = args.lr_ae
        if hasattr(args, 'lr_pretrain') and args.lr_pretrain is not None:
            config.LEARNING_RATE_PRETRAIN = args.lr_pretrain
        if hasattr(args, 'lr_finetune') and args.lr_finetune is not None:
            config.LEARNING_RATE_FINETUNE = args.lr_finetune

        if hasattr(args, 'test_split') and args.test_split is not None:
            config.TEST_SPLIT_FRACTION = args.test_split
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            config.BATCH_SIZE = args.batch_size

        if hasattr(args, 'pipeline') and args.pipeline is not None:
            config.PIPELINE_VERSION = args.pipeline
        if hasattr(args, 'eval_filtered') and args.eval_filtered is not None:
            config.EVAL_FILTERED = args.eval_filtered
        if hasattr(args, 'seed') and args.seed is not None:
            config.SEED = args.seed

        # Output
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            config.OUTPUT_ROOT_DIR = args.output_dir
            # derived paths rebuilt below

        # Memory
        if hasattr(args, 'basin_batch_size') and args.basin_batch_size is not None:
            config.BASIN_PROCESSING_BATCH_SIZE = args.basin_batch_size

        # AE behavior
        if hasattr(args, 'ae_epochs_per_basin') and args.ae_epochs_per_basin is not None:
            config.AE_EPOCHS_PER_BASIN = args.ae_epochs_per_basin
        if hasattr(args, 'ae_patience') and args.ae_patience is not None:
            config.AE_PATIENCE = args.ae_patience

        # ---------- Loss function overrides ----------
        # Legacy support: if LOSS_FUNCTION is set, use it for all stages
        if hasattr(args, 'loss_function') and args.loss_function is not None:
            config.LOSS_FUNCTION = args.loss_function
            config.LOSS_FUNCTION_AE = args.loss_function
            config.LOSS_FUNCTION_PRETRAIN = args.loss_function
            config.LOSS_FUNCTION_FINETUNE = args.loss_function

        # Stage-specific overrides (take precedence)
        if hasattr(args, 'loss_function_ae') and args.loss_function_ae is not None:
            config.LOSS_FUNCTION_AE = args.loss_function_ae
        if hasattr(args, 'loss_function_pretrain') and args.loss_function_pretrain is not None:
            config.LOSS_FUNCTION_PRETRAIN = args.loss_function_pretrain
        if hasattr(args, 'loss_function_finetune') and args.loss_function_finetune is not None:
            config.LOSS_FUNCTION_FINETUNE = args.loss_function_finetune

        if hasattr(args, 'nse_epsilon') and args.nse_epsilon is not None:
            config.NSE_EPSILON = args.nse_epsilon
        if hasattr(args, 'combined_loss_alpha') and args.combined_loss_alpha is not None:
            config.COMBINED_LOSS_ALPHA = args.combined_loss_alpha

        # ---------- Latent save/load ----------
        if hasattr(args, 'save_latents') and args.save_latents:
            config.SAVE_LATENTS = True
        if hasattr(args, 'load_latents') and args.load_latents is not None:
            config.LOAD_LATENTS = True
            config.LATENTS_LOAD_PATH = args.load_latents
        if hasattr(args, 'latents_save_dir') and args.latents_save_dir is not None:
            config.LATENTS_SAVE_DIR = args.latents_save_dir

        # ---------- Autoencoder reconstruction evaluation ----------
        if hasattr(args, 'eval_ae_reconstruction') and args.eval_ae_reconstruction:
            config.EVAL_AE_RECONSTRUCTION = True
        if hasattr(args, 'ae_recon_eval_space') and args.ae_recon_eval_space is not None:
            config.AE_RECON_EVAL_SPACE = args.ae_recon_eval_space
        if hasattr(args, 'save_ae_reconstructions') and args.save_ae_reconstructions:
            config.SAVE_AE_RECONSTRUCTIONS = True
        if hasattr(args, 'ae_recon_max_samples') and args.ae_recon_max_samples is not None:
            config.AE_RECON_MAX_SAMPLES = args.ae_recon_max_samples
        if hasattr(args, 'ae_recon_metrics_filename') and args.ae_recon_metrics_filename is not None:
            config.AE_RECON_METRICS_FILENAME = args.ae_recon_metrics_filename
        if hasattr(args, 'save_per_basin_ae_models') and args.save_per_basin_ae_models:
            config.SAVE_PER_BASIN_AE_MODELS = True

        # ---------- Model version control ----------
        if hasattr(args, 'use_model_v2') and args.use_model_v2:
            config.USE_MODEL_V2 = True
        if hasattr(args, 'use_model_v1') and args.use_model_v1:
            config.USE_MODEL_V2 = False

        # Finalize derived paths and validate
        config._finalize_paths()
        return config

    def print_config(self):
        print(f"\n{'='*70}")
        print("📋 CONFIGURATION SUMMARY")
        print(f"{'='*70}")
        print("\n📁 PATHS:")
        print(f" CAMELS-SPAT Root: {self.CAMELS_SPAT_ROOT}")
        print(f" Basin List: {self.BASIN_LIST_PATH}")
        print(f" ROI Data: {self.ROI_DATA_PATH}")
        print("\n🏗️ MODEL ARCHITECTURE:")
        print(f" Latent Dimension: {self.LATENT_DIM}")
        print(f" LSTM Units: {self.LSTM_UNITS}")
        print(f" LSTM Layers: {self.LSTM_LAYERS}")
        print(f" GRU Units: {self.GRU_UNITS}")
        print(f" Attention Heads: {self.NUM_ATTENTION_HEADS}")
        print(f" Sequence Length: {self.SEQ_LENGTH}")
        print(f" Dropout Rate: {self.DROPOUT_RATE}")
        print(f" Model Version: {'V2 (proper weight transfer)' if self.USE_MODEL_V2 else 'V1 (dual-branch)'}")
        print(f" Shared Autoencoder: {'✅ Enabled' if self.USE_SHARED_AUTOENCODER else '❌ Disabled (per-basin)'}")
        print("\n🎯 TRAINING HYPERPARAMETERS:")
        print(f" Batch Size: {self.BATCH_SIZE}")
        print(f" Epochs (AE): {self.EPOCHS_AE}")
        print(f" Epochs (Pretrain): {self.EPOCHS_PRETRAIN}")
        print(f" Epochs (Finetune): {self.EPOCHS_FINETUNE}")
        print(f" Learning Rate (AE): {self.LEARNING_RATE_AE}")
        print(f" Learning Rate (Pre): {self.LEARNING_RATE_PRETRAIN}")
        print(f" Learning Rate (FT): {self.LEARNING_RATE_FINETUNE}")
        print(f" Test Split: {self.TEST_SPLIT_FRACTION:.1%}")
        print("\n🔧 FEATURE ENGINEERING:")
        print(f" Seasonal Months: {self.SEASONAL_MONTHS}")
        print(f" Month Encoding: {self.MONTH_ENCODING}")
        print(f" Month Embed Dim: {self.MONTH_EMB_DIM}")
        print(f" Flow History: {'✅ Enabled' if self.INCLUDE_FLOW_HISTORY else '❌ Disabled'}")
        if self.INCLUDE_FLOW_HISTORY:
            print(f" Flow History Lag Gap: {self.FLOW_HISTORY_LAG_GAP} timestep(s)")
        print("\n📊 LOSS FUNCTIONS:")
        print(f" Autoencoder: {self.LOSS_FUNCTION_AE}")
        print(f" LSTM Pretrain: {self.LOSS_FUNCTION_PRETRAIN}")
        print(f" Fine-tuning: {self.LOSS_FUNCTION_FINETUNE}")
        print(f" NSE Epsilon: {self.NSE_EPSILON}")
        print(f" Combined Alpha: {self.COMBINED_LOSS_ALPHA}")
        print("\n💾 OUTPUT:")
        print(f" Pretrain Dir: {self.PRETRAIN_OUTPUT_DIR}")
        print(f" Finetune Dir: {self.FINETUNE_OUTPUT_DIR}")
        print("\n🔬 LATENT REPRESENTATIONS:")
        print(f" Save Latents: {'✅ Enabled' if self.SAVE_LATENTS else '❌ Disabled'}")
        if self.SAVE_LATENTS:
            save_dir = self.LATENTS_SAVE_DIR if self.LATENTS_SAVE_DIR else f"{self.PRETRAIN_OUTPUT_DIR}/latents"
            print(f" Save Directory: {save_dir}")
        print(f" Load Latents: {'✅ Enabled' if self.LOAD_LATENTS else '❌ Disabled'}")
        if self.LOAD_LATENTS:
            print(f" Load Path: {self.LATENTS_LOAD_PATH}")
        print("\n🔀 PIPELINE:")
        print(f" Version: {self.PIPELINE_VERSION}")
        print(f" Eval Filtered: {self.EVAL_FILTERED}")
        print(f" Random Seed: {self.SEED}")
        print(f"\n\U0001f9ea LATENT REPRESENTATIONS:")
        print(f" Save Latents: {'✅ Enabled' if self.SAVE_LATENTS else '❌ Disabled'}")
        if self.SAVE_LATENTS:
            save_dir = self.LATENTS_SAVE_DIR if self.LATENTS_SAVE_DIR else f"{self.PRETRAIN_OUTPUT_DIR}/latents"
            print(f" Save Directory: {save_dir}")
        print(f" Load Latents: {'✅ Enabled' if self.LOAD_LATENTS else '❌ Disabled'}")
        if self.LOAD_LATENTS:
            print(f" Load Path: {self.LATENTS_LOAD_PATH}")
        print("\n\U0001f4c9 AUTOENCODER RECONSTRUCTION EVAL:")
        ae_eval_status = "✅ Enabled" if self.EVAL_AE_RECONSTRUCTION else "❌ Disabled"
        print(f" AE Recon Eval: {ae_eval_status}")
        if self.EVAL_AE_RECONSTRUCTION:
            print(f" Recon Eval Space: {self.AE_RECON_EVAL_SPACE}")
            save_recon_status = "✅ Enabled" if self.SAVE_AE_RECONSTRUCTIONS else "❌ Disabled"
            print(f" Save Reconstructions: {save_recon_status}")
            print(f" Recon Max Samples: {self.AE_RECON_MAX_SAMPLES}")
            print(f" Metrics File: {self.AE_RECON_METRICS_FILENAME}")
        save_per_basin_status = "✅ Enabled" if self.SAVE_PER_BASIN_AE_MODELS else "❌ Disabled"
        print(f" Save Per-Basin AE Models: {save_per_basin_status}")
        print(f"\n{'='*70}\n")

    def validate(self):
        """
        Validate configuration parameters for consistency and correctness.
        Raises ValueError if any configuration is invalid.
        """
        errors = []

        # Validate month embedding dimension for sinusoidal encoding
        if self.MONTH_ENCODING == 'sinusoidal' and self.MONTH_EMB_DIM % 2 != 0:
            errors.append("MONTH_EMB_DIM must be even for sinusoidal encoding")

        # Validate sequence length
        if self.SEQ_LENGTH < 1:
            errors.append("SEQ_LENGTH must be at least 1")

        # Validate predict ahead
        if self.PREDICT_AHEAD < 1:
            errors.append("PREDICT_AHEAD must be at least 1")

        # Validate flow history lag gap
        if self.FLOW_HISTORY_LAG_GAP < 0:
            errors.append("FLOW_HISTORY_LAG_GAP must be non-negative")

        if self.FLOW_HISTORY_LAG_GAP >= self.SEQ_LENGTH:
            errors.append("FLOW_HISTORY_LAG_GAP must be less than SEQ_LENGTH")

        # Validate the test split fraction
        if not 0 < self.TEST_SPLIT_FRACTION < 1:
            errors.append("TEST_SPLIT_FRACTION must be between 0 and 1")

        # Validate dropout rate
        if not 0 <= self.DROPOUT_RATE < 1:
            errors.append("DROPOUT_RATE must be between 0 and 1")

        # Validate learning rates
        for lr_name in ['LEARNING_RATE_AE', 'LEARNING_RATE_PRETRAIN', 'LEARNING_RATE_FINETUNE']:
            lr = getattr(self, lr_name)
            if lr <= 0:
                errors.append(f"{lr_name} must be positive")

        # Validate dimensions
        if self.LATENT_DIM < 1:
            errors.append("LATENT_DIM must be at least 1")
        if self.LSTM_UNITS < 1:
            errors.append("LSTM_UNITS must be at least 1")
        if self.GRU_UNITS < 1:
            errors.append("GRU_UNITS must be at least 1")
        if self.NUM_ATTENTION_HEADS < 1:
            errors.append("NUM_ATTENTION_HEADS must be at least 1")

        # Validate AE recon eval space
        if self.AE_RECON_EVAL_SPACE not in {'scaled', 'original', 'both'}:
            errors.append("AE_RECON_EVAL_SPACE must be one of: 'scaled', 'original', 'both'")

        if self.AE_RECON_MAX_SAMPLES is not None and self.AE_RECON_MAX_SAMPLES <= 0:
            errors.append("AE_RECON_MAX_SAMPLES must be a positive integer or None")

        # Validate seasonal sequence mode
        if self.SEASONAL_SEQUENCE_MODE not in {'contiguous', 'skip'}:
            errors.append("SEASONAL_SEQUENCE_MODE must be 'contiguous' or 'skip'")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        print("✅ Configuration validation passed")
