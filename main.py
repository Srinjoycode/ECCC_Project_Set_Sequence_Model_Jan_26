import argparse
import os
import tensorflow as tf
from config import Config

# Import the new pipeline
from training_pipelines.set_sequence_pipeline import pipeline_set_sequence

from utils.seed_utils import set_global_seed
from utils.set_seq_data_loading import load_basin_list


def main():
    parser = argparse.ArgumentParser(description="Run Hydrological Deep Learning Pipelines")
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON')
    parser.add_argument('--pipeline', type=str, choices=['set_seq'], required=True,
                        help='Pipeline version to run')

    # Replaced boolean flag with string argument for filtering mode
    parser.add_argument('--filtering-mode', type=str, choices=['none', 'eval_only', 'finetune', 'all'],
                        default=None, help='Override filtering mode from config')

    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"✅ Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️  No GPU found, running on CPU")

    # Load Config
    config = Config.from_args(args)
    set_global_seed(config.SEED)

    # Load Basin List
    basin_list_path = config.BASIN_LIST_PATH
    if not os.path.exists(basin_list_path):
        raise FileNotFoundError(f"Basin list not found: {basin_list_path}")

    basin_list = load_basin_list(basin_list_path)

    # Route to Pipeline
    if args.pipeline == 'set_seq':
        # New Set-Sequence Pipeline
        results = pipeline_set_sequence(config, basin_list, args)

    print("\n✅ Execution Complete.")
    print(results)


if __name__ == "__main__":
    main()