# ==========================================
# SET-SEQUENCE CONFIGURATION
# ==========================================
import os, json

class Config:
    """Configuration tailored to the set-sequence pipeline."""

    def __init__(self, config_file=None):
        # Paths
        self.CAMELS_SPAT_ROOT = "/media/sbhuiya/1a899d3a-b2a4-487c-b59c-fd2cac4442c8/CAMELS-SPAT"
        self.BASIN_LIST_PATH = "basin_lists/camels_pretraining_basin_list_short.txt"
        self.ROI_DATA_PATH = "FinalData/Dataset4.csv"
        self.OUTPUT_ROOT_DIR = "Results/set_seq_"
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
        self.STEPS_PER_EPOCH = 1000
        self.VAL_STEPS = 200

        # Features
        self.SEASONAL_MONTHS = [5, 6, 7, 8, 9]
        self.MONTH_ENCODING = 'sinusoidal'
        self.MONTH_EMB_DIM = 12

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
            'steps_per_epoch': self.STEPS_PER_EPOCH,
            'val_steps': self.VAL_STEPS,
            'seasonal_months': self.SEASONAL_MONTHS,
            'month_encoding': self.MONTH_ENCODING,
            'month_emb_dim': self.MONTH_EMB_DIM,
            'seed': self.SEED,
        }

    @classmethod
    def from_args(cls, args):
        config = cls(args.config_file) if getattr(args, 'config_file', None) else cls()

        if hasattr(args, 'basin_list') and args.basin_list is not None:
            config.BASIN_LIST_PATH = args.basin_list
        if hasattr(args, 'roi_data_path') and args.roi_data_path is not None:
            config.ROI_DATA_PATH = args.roi_data_path
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            config.OUTPUT_ROOT_DIR = args.output_dir
        if hasattr(args, 'pretrain_output_dir') and args.pretrain_output_dir is not None:
            config.PRETRAIN_OUTPUT_DIR = args.pretrain_output_dir

        if hasattr(args, 'latent_dim') and args.latent_dim is not None:
            config.LATENT_DIM = args.latent_dim
        if hasattr(args, 'lstm_units') and args.lstm_units is not None:
            config.LSTM_UNITS = args.lstm_units
        if hasattr(args, 'lstm_layers') and args.lstm_layers is not None:
            config.LSTM_LAYERS = args.lstm_layers
        if hasattr(args, 'seq_length') and args.seq_length is not None:
            config.SEQ_LENGTH = args.seq_length
        if hasattr(args, 'predict_ahead') and args.predict_ahead is not None:
            config.PREDICT_AHEAD = args.predict_ahead
        if hasattr(args, 'dropout_rate') and args.dropout_rate is not None:
            config.DROPOUT_RATE = args.dropout_rate

        if hasattr(args, 'batch_size') and args.batch_size is not None:
            config.BATCH_SIZE = args.batch_size
        if hasattr(args, 'epochs_pretrain') and args.epochs_pretrain is not None:
            config.EPOCHS_PRETRAIN = args.epochs_pretrain
        if hasattr(args, 'epochs_finetune') and args.epochs_finetune is not None:
            config.EPOCHS_FINETUNE = args.epochs_finetune
        if hasattr(args, 'lr_pretrain') and args.lr_pretrain is not None:
            config.LEARNING_RATE_PRETRAIN = args.lr_pretrain
        if hasattr(args, 'lr_finetune') and args.lr_finetune is not None:
            config.LEARNING_RATE_FINETUNE = args.lr_finetune
        if hasattr(args, 'test_split') and args.test_split is not None:
            config.TEST_SPLIT_FRACTION = args.test_split
        if hasattr(args, 'steps_per_epoch') and args.steps_per_epoch is not None:
            config.STEPS_PER_EPOCH = args.steps_per_epoch
        if hasattr(args, 'val_steps') and args.val_steps is not None:
            config.VAL_STEPS = args.val_steps

        if hasattr(args, 'month_encoding') and args.month_encoding is not None:
            config.MONTH_ENCODING = args.month_encoding
        if hasattr(args, 'month_emb_dim') and args.month_emb_dim is not None:
            config.MONTH_EMB_DIM = args.month_emb_dim
        if hasattr(args, 'seed') and args.seed is not None:
            config.SEED = args.seed

        config._finalize_paths()
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
        print(f"  Steps/Epoch      : {self.STEPS_PER_EPOCH}")
        print(f"  Val Steps        : {self.VAL_STEPS}")
        print("FEATURES:")
        print(f"  Seasonal Months  : {self.SEASONAL_MONTHS}")
        print(f"  Month Encoding   : {self.MONTH_ENCODING}")
        print(f"  Month Emb Dim    : {self.MONTH_EMB_DIM}")
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
        for name in ['LATENT_DIM', 'LSTM_UNITS', 'LSTM_LAYERS', 'BATCH_SIZE', 'EPOCHS_PRETRAIN', 'EPOCHS_FINETUNE', 'STEPS_PER_EPOCH', 'VAL_STEPS']:
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        for name in ['LEARNING_RATE_PRETRAIN', 'LEARNING_RATE_FINETUNE']:
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if self.MONTH_ENCODING == 'sinusoidal' and self.MONTH_EMB_DIM % 2 != 0:
            errors.append("MONTH_EMB_DIM must be even for sinusoidal encoding")
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        print("✅ Configuration validation passed")
