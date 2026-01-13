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

        potential_vars = ['prcp', 'tmin', 'tmax', 'tavg'] #['prcp', 'tmin', 'tmax', 'tavg', 'swe', 'vp', 'dayl']

        available_vars = [v for v in potential_vars if v in forcing_ds.data_vars]

        if not available_vars:
            forcing_ds.close()
            flow_ds.close()
            return None, None, None, None

        print("Available vars found", available_vars)

        # Resample to monthly
        monthly_forcing = forcing_ds.resample(time='MS').mean()
        monthly_flow = flow_ds.resample(time='MS').mean()

        # --- DATA EXTRACTION ---
        feature_list = []
        for var in available_vars:
            data = monthly_forcing[var].values  # Shape: (Time, Lat, Lon)
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
    print(f"Sample basin shape: {X_list[0].shape}")

    return X_list, M_list, y_list, successful_basins, months_list


# def process_roi_csv(csv_path, apply_seasonal_filter=False, target_months=None, config=None, flatten_spatial=True):
#     """
#     MODIFIED: Handles ROI CSVs.
#     If flatten_spatial=False, reshapes X_meteo to (Time, 1, Features) to act as a "Set of size 1".
#     """
#     print(f"\nProcessing ROI dataset: {csv_path}")
#     df = pd.read_csv(csv_path)
#
#     if apply_seasonal_filter and target_months is not None and 'Month' in df.columns:
#         mask = df['Month'].isin(target_months)
#         df = df[mask].reset_index(drop=True)
#
#     meteo_keywords = ['precip', 'temp', 'SD_', 'SW_']
#     meteo_cols = [c for c in df.columns if any(k in c for k in meteo_keywords)]
#     exclude = ['Flow', 'Year', 'YM', 'Month', 'Unnamed: 0'] + meteo_cols
#     global_cols = [c for c in df.columns if c not in exclude]
#
#     # Month handling (same as original)
#     if 'Month' in df.columns:
#         months = df['Month'].values
#         if config is not None and getattr(config, 'MONTH_ENCODING', 'dummy') == 'sinusoidal':
#             emb_dim = getattr(config, 'MONTH_EMB_DIM', 12)
#             month_dummies = month_sinusoidal_embedding(months, dim=emb_dim, period=12).astype(DEFAULT_DTYPE)
#         else:
#             month_dummies = pd.get_dummies(df['Month'], prefix='Month').reindex(
#                 columns=[f'Month_{i}' for i in range(1, 13)], fill_value=0
#             ).values.astype(DEFAULT_DTYPE)
#     else:
#         month_dummies = np.zeros((len(df), 12), dtype=DEFAULT_DTYPE)
#         months = np.zeros(len(df))
#
#     X_meteo = df[meteo_cols].values.astype(DEFAULT_DTYPE)
#
#     if not flatten_spatial:
#         # Reshape to (Time, 1, Features) for Set-Sequence model
#         X_meteo = X_meteo[:, np.newaxis, :]
#
#     X_global_indices = df[global_cols].values.astype(DEFAULT_DTYPE) if global_cols else np.zeros((len(df), 1),
#                                                                                                  dtype=DEFAULT_DTYPE)
#     X_global = np.hstack([month_dummies, X_global_indices])
#     y_flow = df['Flow'].values.reshape(-1, 1).astype(DEFAULT_DTYPE)
#
#     print(f" ROI features - Meteo: {X_meteo.shape}, Global: {X_global.shape}, Flow: {y_flow.shape}")
#     return X_meteo, X_global, y_flow, months
#

def process_roi_csv(csv_path, apply_seasonal_filter=False, target_months=None, config=None, flatten_spatial=True):
    print(f"\nProcessing ROI dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Seasonal Filtering
    if apply_seasonal_filter and target_months is not None and 'Month' in df.columns:
        mask = df['Month'].isin(target_months)
        df = df[mask].reset_index(drop=True)

    # 2. Define the stations and variables to extract
    # Based on your CSV header: Banff, Lake Louise, Calgary Intl A
    stations = ['Lake Louise', 'Calgary Intl A', 'Banff',"Elbow Ranger Station"]

    # Map model features (generic) to CSV columns (specific)
    # The order MUST match 'potential_vars' in process_netcdf_basin: ['prcp', 'tmin', 'tmax', 'tavg']
    var_map = {
        'prcp': '_precip',
        'tmin': '_min_temp',
        'tmax': '_max_temp',
        'tavg': '_avetemp_avg_temp'
    }

    # 3. Build the 3D Set Tensor: (Time, Stations, Variables)
    X_set_list = []

    for station in stations:
        station_feats = []
        is_station_valid = True

        for model_var, csv_suffix in var_map.items():
            col_name = f"{station}{csv_suffix}"

            if col_name in df.columns:
                station_feats.append(df[col_name].values)
            else:
                # Handle missing data (e.g., Banff might lack temp)
                # For Set-Sequence, we can fill with 0 or skip the station.
                # Here we fill with 0 to keep the station in the set.
                # print(f"Warning: {col_name} missing. Filling 0.")
                station_feats.append(np.zeros(len(df)))

                # If you prefer to drop stations with missing data, uncomment below:
                # is_station_valid = False
                # break

        if is_station_valid:
            # Stack variables for this station -> (Time, N_Vars)
            station_data = np.stack(station_feats, axis=-1)
            X_set_list.append(station_data)

    # Stack stations -> (Time, N_Stations, N_Vars)
    # Result shape: (Time, 3, 4)
    X_meteo = np.stack(X_set_list, axis=1).astype(DEFAULT_DTYPE)

    if flatten_spatial:
        # Fallback for old pipelines (Time, N_Stations * N_Vars)
        X_meteo = X_meteo.reshape(X_meteo.shape[0], -1)

    # 4. Global Features (Indices) & Month
    # Exclude the station columns we just processed plus meta columns
    used_cols = [f"{s}{v}" for s in stations for v in var_map.values()]
    meta_cols = ['Year', 'Month', 'YM', 'Flow', 'Unnamed: 0']
    exclude = used_cols + meta_cols

    global_cols = [c for c in df.columns if c not in exclude and c in df.columns]

    # Month dummies
    if 'Month' in df.columns:
        months = df['Month'].values
        if config is not None and getattr(config, 'MONTH_ENCODING', 'dummy') == 'sinusoidal':
            emb_dim = getattr(config, 'MONTH_EMB_DIM', 12)
            month_dummies = month_sinusoidal_embedding(months, dim=emb_dim, period=12).astype(DEFAULT_DTYPE)
        else:
            month_dummies = pd.get_dummies(df['Month'], prefix='Month').reindex(
                columns=[f'Month_{i}' for i in range(1, 13)], fill_value=0
            ).values.astype(DEFAULT_DTYPE)
    else:
        month_dummies = np.zeros((len(df), 12), dtype=DEFAULT_DTYPE)
        months = np.zeros(len(df))

    X_global_indices = df[global_cols].values.astype(DEFAULT_DTYPE) if global_cols else np.zeros((len(df), 1),
                                                                                                 dtype=DEFAULT_DTYPE)
    X_global = np.hstack([month_dummies, X_global_indices])

    y_flow = df['Flow'].values.reshape(-1, 1).astype(DEFAULT_DTYPE)

    print(f" ROI processed. Shape: {X_meteo.shape} (Time, Stations, Vars)")
    print(f" Stations processed: {stations}")

    return X_meteo, X_global, y_flow, months




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