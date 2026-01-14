import gc
import numpy as np
from pathlib import Path
import xarray as xr
import pandas as pd
from utils.month_emb_utils import month_sinusoidal_embedding

DEFAULT_DTYPE = np.float32

def load_basin_list(basin_list_path):
    # (Same as your original)
    basins = []
    with open(basin_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                basins.append(line)
    return basins

def get_basin_scale(basin_id, camels_root):


    meso_path = Path(camels_root) / "observations" / "meso-scale" / "obs-daily" / f"{basin_id}_daily_flow_observations.nc"
    macro_path = Path(camels_root) / "observations" / "macro-scale" / "obs-daily" / f"{basin_id}_daily_flow_observations.nc"

    print(meso_path)
    print(macro_path)

    if meso_path.exists():
        return "meso-scale"
    elif macro_path.exists():
        return "macro-scale"
    else:
        return None


def process_netcdf_basin(basin_id, camels_root, apply_seasonal_filter=False, target_months=None, config=None,
                         flatten_spatial=True):
    """
    MODIFIED: Added flatten_spatial argument.
    - If True (default): Flattens grid into 1D vector (for Autoencoders).
    - If False: Stacks grid into (Time, Grid, Feat) (for Set-Sequence).
    """
    try:
        print("Entering process_netcdf_basin with basin id ", basin_id)
        scale = get_basin_scale(basin_id, camels_root)


        print("Basin Scale found", scale)
        if scale is None:
            return None, None, None, None

        forcing_path = Path(camels_root) / "daymet-distributed" / f"{basin_id}_daymet_distributed.nc"
        flow_path = Path(camels_root) / "observations" / scale / "obs-daily" / f"{basin_id}_daily_flow_observations.nc"

        print("Paths found")
        print(forcing_path)
        print(flow_path)

        if not forcing_path.exists() or not flow_path.exists():
            return None, None, None, None

        forcing_ds = xr.open_dataset(forcing_path)
        flow_ds = xr.open_dataset(flow_path)
        forcing_ds['tavg'] = (forcing_ds['tmin'] + forcing_ds['tmax']) / 2

        print("Datasets opened")

        if not forcing_ds.data_vars:
            print(f"No data variables found in {forcing_path}")
            forcing_ds.close()
            flow_ds.close()
            return None, None, None, None

        # Standardize features to ensure consistency across basins and with ROI
        required_vars = ['prcp', 'tmin', 'tmax', 'tavg']
        # potential_vars = required_vars (redundant, we iterate required_vars)

        # Resample to monthly
        monthly_forcing = forcing_ds.resample(time='MS').mean()
        monthly_flow = flow_ds.resample(time='MS').mean()

        # --- DATA EXTRACTION ---
        feature_list = []
        for var in required_vars:
            if var in monthly_forcing.data_vars:
                data = monthly_forcing[var].values  # Shape: (Time, Lat, Lon)
            else:
                # Fill zeros if missing
                # Use ANY present variable to determine shape
                ref_var = list(forcing_ds.data_vars)[0]
                data = np.zeros_like(monthly_forcing[ref_var].values)

            data = np.nan_to_num(data, nan=0.0)

            # Reshape spatial dims (Lat, Lon) -> (Grid_Points)
            # data_reshaped: (Time, Grid_Points) if flattened later, or kept as (Time, Grid_Points, 1) if stacked
            data_reshaped = data.reshape(data.shape[0], -1)

            if not flatten_spatial:
                # Expand dims to (Time, Grid, 1) so we can concatenate features along last axis
                data_reshaped = data_reshaped[..., np.newaxis]

            feature_list.append(data_reshaped)

        if flatten_spatial:
            # ORIGINAL BEHAVIOR: (Time, Grid * Vars)
            X_processed = np.hstack(feature_list).astype(DEFAULT_DTYPE)
        else:
            # SET-SEQUENCE BEHAVIOR: (Time, Grid, Vars)
            X_processed = np.concatenate(feature_list, axis=-1).astype(DEFAULT_DTYPE)

        # Flow targets
        flow_vars = list(monthly_flow.data_vars)
        y_flow = monthly_flow[flow_vars[0]].values.reshape(-1, 1).astype(DEFAULT_DTYPE)
        y_flow = np.nan_to_num(y_flow, nan=0.0)

        # Sync lengths
        min_len = min(len(X_processed), len(y_flow))
        X_processed = X_processed[:min_len]
        y_flow = y_flow[:min_len]

        time_index = monthly_forcing.time.values[:min_len]
        months = np.array(pd.DatetimeIndex(time_index).month)

        if apply_seasonal_filter and target_months is not None:
            mask = np.isin(months, target_months)
            X_processed = X_processed[mask]
            y_flow = y_flow[mask]
            months = months[mask]

        # Month embedding (same as original)
        if config is not None and getattr(config, 'MONTH_ENCODING', 'dummy') == 'sinusoidal':
            emb_dim = getattr(config, 'MONTH_EMB_DIM', 12)
            month_dummies = month_sinusoidal_embedding(months, dim=emb_dim, period=12).astype(DEFAULT_DTYPE)
        else:
            month_dummies = pd.get_dummies(pd.Series(months), prefix='Month').reindex(
                columns=[f'Month_{i}' for i in range(1, 13)], fill_value=0
            ).values.astype(DEFAULT_DTYPE)

        forcing_ds.close()
        flow_ds.close()
        return X_processed, month_dummies, y_flow, months

    except Exception as e:
        print(f"Error processing basin {basin_id}: {e}")
        return None, None, None, None


def load_all_pretrain_basins(basin_list, camels_root, apply_seasonal_filter=False, target_months=None, config=None,
                             flatten_spatial=True):
    """
    MODIFIED: Accepts flatten_spatial to pass down to process_netcdf_basin.
    """
    X_list, M_list, y_list = [], [], []
    months_list = []
    successful_basins = []

    print(f"\n{'=' * 60}")
    print(f"Loading {len(basin_list)} pretraining basins (Flatten Spatial: {flatten_spatial})...")

    batch_size = getattr(config, 'BASIN_PROCESSING_BATCH_SIZE', 50) if config is not None else 50

    for i, basin_id in enumerate(basin_list):
        print("Trying to load for basin :", basin_id)

        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(basin_list)} basins processed...")
            gc.collect()

        X, M, y, months = process_netcdf_basin(
            basin_id, camels_root, apply_seasonal_filter, target_months, config=config, flatten_spatial=flatten_spatial
        )
        print("Loaded basin for basin :", basin_id)

        if X is not None and len(X) > 0:
            X_list.append(X.astype(DEFAULT_DTYPE))
            M_list.append(M.astype(DEFAULT_DTYPE))
            y_list.append(y.astype(DEFAULT_DTYPE))
            months_list.append(months)
            successful_basins.append(basin_id)

    print(f"\nSuccessfully loaded {len(successful_basins)}/{len(basin_list)} basins")
    if not X_list:
        raise ValueError("No valid basins were loaded!")

    # Check shape of first loaded basin
    if len(X_list) > 0:
        print(f"Sample basin shape: {X_list[0].shape}")

    return X_list, M_list, y_list, successful_basins, months_list


def process_roi_csv(csv_path, apply_seasonal_filter=False, target_months=None, config=None, flatten_spatial=True):
    """
    MODIFIED: Handles ROI CSVs.
    Parses columns to extract stations and creates a spatial 'Set' structure.
    Returns X as (Time, Grid, 4) where 4 = [prcp, tmin, tmax, tavg].
    """
    print(f"\nProcessing ROI dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if apply_seasonal_filter and target_months is not None and 'Month' in df.columns:
        mask = df['Month'].isin(target_months)
        df = df[mask].reset_index(drop=True)

    # 1. Identify Stations and their columns
    # Heuristic: <StationName>_<VariableSuffix>
    # We look for columns ending in the known suffixes for the 4 required vars.
    var_map = {
        'prcp': '_precip',
        'tmin': '_min_temp',
        'tmax': '_max_temp',
        'tavg': '_avetemp_avg_temp'
    }
    
    # Find all potential stations
    all_cols = df.columns.tolist()
    stations = set()
    for col in all_cols:
        for var, suffix in var_map.items():
            if col.endswith(suffix):
                stations.add(col[:-len(suffix)])
    
    stations = sorted(list(stations))
    print(f"Identified {len(stations)} stations in ROI: {stations}")
    
    if not stations:
        raise ValueError("No stations identified in ROI CSV! Check column suffixes.")

    # 2. Build 3D Tensor (Time, Stations, 4)
    n_time = len(df)
    n_stations = len(stations)
    n_features = 4  # Fixed to [prcp, tmin, tmax, tavg]
    
    X_spatial = np.zeros((n_time, n_stations, n_features), dtype=np.float32)
    
    for i, station in enumerate(stations):
        for j, var_key in enumerate(['prcp', 'tmin', 'tmax', 'tavg']):
            suffix = var_map[var_key]
            col_name = f"{station}{suffix}"
            if col_name in df.columns:
                X_spatial[:, i, j] = df[col_name].values

    if flatten_spatial:
        X_processed = X_spatial.reshape(n_time, -1)
    else:
        X_processed = X_spatial

    # 3. Targets and Others
    y_flow = df['Flow'].values.reshape(-1, 1).astype(np.float32) if 'Flow' in df.columns else np.zeros((n_time, 1), dtype=np.float32)
    roi_months = df['Month'].values if 'Month' in df.columns else None
    X_global = None 

    print(f"ROI Processed Shape: {X_processed.shape}")
    return X_processed, X_global, y_flow, roi_months


def temporal_train_test_split(X, y, test_fraction, gap_fraction=0.0, months=None):
    n = len(y)
    train_end = int(n * (1 - test_fraction - gap_fraction))
    test_start = int(n * (1 - test_fraction))

    if isinstance(X, list):
        X_train = [x[:train_end] for x in X]
        X_test = [x[test_start:] for x in X]
    else:
        X_train = X[:train_end]
        X_test = X[test_start:]

    y_train = y[:train_end]
    y_test = y[test_start:]

    if months is not None:
        months_train = months[:train_end]
        months_test = months[test_start:]
        return X_train, X_test, y_train, y_test, months_train, months_test

    return X_train, X_test, y_train, y_test