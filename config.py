# ==========================================
# SET-SEQUENCE CONFIGURATION
# ==========================================
import os, json

class Config:
    """Configuration tailored to the set-sequence pipeline."""

    def __init__(self, config_file=None):
        # Paths
        self.CAMELS_SPAT_ROOT = "O:\CAMELS_SPAT"
        self.BASIN_LIST_PATH = "basin_lists/camels_pretraining_basin_list_short.txt"
        self.ROI_DATA_PATH = "FinalData/Dataset4.csv"
        self.OUTPUT_ROOT_DIR = "Results/test_1"
        self.PRETRAIN_OUTPUT_DIR = None  # derived if None

        # Model
        self.LATENT_DIM = 512
        self.LSTM_UNITS = 512
        self.LSTM_LAYERS = 2
        self.SEQ_LENGTH = 12
        self.PREDICT_AHEAD = 1
        self.DROPOUT_RATE = 0.35

        # Training
        self.BATCH_SIZE = 512
        self.EPOCHS_PRETRAIN = 300
        self.EPOCHS_FINETUNE = 200
        self.LEARNING_RATE_PRETRAIN = 5e-4
        self.LEARNING_RATE_FINETUNE = 5e-4
        self.TEST_SPLIT_FRACTION = 0.23

        # Patience settings
        self.PATIENCE_PRETRAIN = 50
        self.PATIENCE_FINETUNE = 15
        
        # LR Scheduler settings
        self.LR_FACTOR = 0.5
        self.LR_PATIENCE = 15

        # Features & Filtering
        self.SEASONAL_MONTHS = [5, 6, 7, 8, 9]
        self.MONTH_ENCODING = 'sinusoidal'
        self.MONTH_EMB_DIM = 12
        
        # Filtering Mode: 'none', 'eval_only', 'finetune', 'all'
        self.FILTERING_MODE = 'eval_only'
        
        # ROI Station Selection: 'all' (zero-fill missing), 'complete_only' (drop incomplete stations)
        self.ROI_STATION_SELECTION = 'complete_only'

        # Reproducibility
        self.SEED = 42

        if config_file is not None:
            self.load_from_file(config_file)

        self._finalize_paths()

    def _finalize_paths(self):
        if self.PRETRAIN_OUTPUT_DIR is None:
            self.PRETRAIN_OUTPUT_DIR = os.path.join(self.OUTPUT_ROOT_DIR, 'pretraining')

    def load_from_file(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Map JSON keys to class attributes (case-insensitive)
        for key, value in config_dict.items():
            attr_name = key.upper()
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)
            else:
                # Handle specific mappings for legacy or mismatched keys
                if key == "pretrain_output_dir":
                    self.PRETRAIN_OUTPUT_DIR = value
                elif key == "output_root_dir":
                    self.OUTPUT_ROOT_DIR = value
                elif key == "roi_data_path":
                    self.ROI_DATA_PATH = value
                elif key == "basin_list_path":
                    self.BASIN_LIST_PATH = value
                elif key == "camels_spat_root":
                    self.CAMELS_SPAT_ROOT = value
                # Ignore steps_per_epoch / val_steps if present in JSON
                elif key.lower() in ["steps_per_epoch", "val_steps"]:
                    pass 
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
            'camels_spat_root': self.CAMELS_SPAT_ROOT,
            'basin_list_path': self.BASIN_LIST_PATH,
            'roi_data_path': self.ROI_DATA_PATH,
            'output_root_dir': self.OUTPUT_ROOT_DIR,
            'pretrain_output_dir': self.PRETRAIN_OUTPUT_DIR,
            'latent_dim': self.LATENT_DIM,
            'lstm_units': self.LSTM_UNITS,
            'lstm_layers': self.LSTM_LAYERS,
            'seq_length': self.SEQ_LENGTH,
            'predict_ahead': self.PREDICT_AHEAD,
            'dropout_rate': self.DROPOUT_RATE,
            'batch_size': self.BATCH_SIZE,
            'epochs_pretrain': self.EPOCHS_PRETRAIN,
            'epochs_finetune': self.EPOCHS_FINETUNE,
            'learning_rate_pretrain': self.LEARNING_RATE_PRETRAIN,
            'learning_rate_finetune': self.LEARNING_RATE_FINETUNE,
            'test_split_fraction': self.TEST_SPLIT_FRACTION,
            'patience_pretrain': self.PATIENCE_PRETRAIN,
            'patience_finetune': self.PATIENCE_FINETUNE,
            'lr_factor': self.LR_FACTOR,
            'lr_patience': self.LR_PATIENCE,
            'seasonal_months': self.SEASONAL_MONTHS,
            'month_encoding': self.MONTH_ENCODING,
            'month_emb_dim': self.MONTH_EMB_DIM,
            'filtering_mode': self.FILTERING_MODE,
            'roi_station_selection': self.ROI_STATION_SELECTION,
            'seed': self.SEED,
        }

    @classmethod
    def from_args(cls, args):
        config = cls(args.config) if hasattr(args, 'config') and args.config else cls()

        # Override with command line arguments if present
        if hasattr(args, 'gpu'):
            # GPU is handled in main, but we can store it if needed
            pass
        
        if hasattr(args, 'filtering_mode') and args.filtering_mode:
            config.FILTERING_MODE = args.filtering_mode

        return config

    def print_config(self):
        print(f"\n{'='*70}")
        print("📋 SET-SEQUENCE CONFIG")
        print(f"{'='*70}")
        print("PATHS:")
        print(f"  CAMELS-SPAT Root : {self.CAMELS_SPAT_ROOT}")
        print(f"  Basin List       : {self.BASIN_LIST_PATH}")
        print(f"  ROI Data         : {self.ROI_DATA_PATH}")
        print(f"  Output Root      : {self.OUTPUT_ROOT_DIR}")
        print(f"  Pretrain Output  : {self.PRETRAIN_OUTPUT_DIR}")
        print("MODEL:")
        print(f"  Latent Dim       : {self.LATENT_DIM}")
        print(f"  LSTM Units       : {self.LSTM_UNITS}")
        print(f"  LSTM Layers      : {self.LSTM_LAYERS}")
        print(f"  Sequence Length  : {self.SEQ_LENGTH}")
        print(f"  Predict Ahead    : {self.PREDICT_AHEAD}")
        print(f"  Dropout Rate     : {self.DROPOUT_RATE}")
        print("TRAINING:")
        print(f"  Batch Size       : {self.BATCH_SIZE}")
        print(f"  Epochs Pretrain  : {self.EPOCHS_PRETRAIN}")
        print(f"  Epochs Finetune  : {self.EPOCHS_FINETUNE}")
        print(f"  LR Pretrain      : {self.LEARNING_RATE_PRETRAIN}")
        print(f"  LR Finetune      : {self.LEARNING_RATE_FINETUNE}")
        print(f"  Test Split       : {self.TEST_SPLIT_FRACTION:.2f}")
        print(f"  Patience Pretrain: {self.PATIENCE_PRETRAIN}")
        print(f"  Patience Finetune: {self.PATIENCE_FINETUNE}")
        print(f"  LR Factor        : {self.LR_FACTOR}")
        print(f"  LR Patience      : {self.LR_PATIENCE}")
        print("FEATURES & FILTERING:")
        print(f"  Seasonal Months  : {self.SEASONAL_MONTHS}")
        print(f"  Month Encoding   : {self.MONTH_ENCODING}")
        print(f"  Month Emb Dim    : {self.MONTH_EMB_DIM}")
        print(f"  Filtering Mode   : {self.FILTERING_MODE}")
        print(f"  ROI Station Sel  : {self.ROI_STATION_SELECTION}")
        print(f"SEED: {self.SEED}")
        print(f"{'='*70}\n")

    def validate(self):
        errors = []
        if self.SEQ_LENGTH < 1:
            errors.append("SEQ_LENGTH must be at least 1")
        if self.PREDICT_AHEAD < 1:
            errors.append("PREDICT_AHEAD must be at least 1")
        if not 0 < self.TEST_SPLIT_FRACTION < 1:
            errors.append("TEST_SPLIT_FRACTION must be between 0 and 1")
        if not 0 <= self.DROPOUT_RATE < 1:
            errors.append("DROPOUT_RATE must be between 0 and 1")
        for name in ['LATENT_DIM', 'LSTM_UNITS', 'LSTM_LAYERS', 'BATCH_SIZE', 'EPOCHS_PRETRAIN', 'EPOCHS_FINETUNE']:
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        for name in ['LEARNING_RATE_PRETRAIN', 'LEARNING_RATE_FINETUNE']:
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if self.MONTH_ENCODING == 'sinusoidal' and self.MONTH_EMB_DIM % 2 != 0:
            errors.append("MONTH_EMB_DIM must be even for sinusoidal encoding")
        
        valid_modes = ['none', 'eval_only', 'finetune', 'all']
        if self.FILTERING_MODE not in valid_modes:
            errors.append(f"FILTERING_MODE must be one of {valid_modes}")
            
        valid_roi_sel = ['all', 'complete_only']
        if self.ROI_STATION_SELECTION not in valid_roi_sel:
            errors.append(f"ROI_STATION_SELECTION must be one of {valid_roi_sel}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        print("✅ Configuration validation passed")
