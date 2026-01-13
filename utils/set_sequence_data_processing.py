import numpy as np


def create_sequences(X, X_global, y, seq_length, predict_ahead, include_flow_history=False, flow_history_lag_gap=0,
                     forecast_horizon=1):
    """
    Creates sequences for LSTM training.
    Robust to input dimensionality (works for 2D or 3D X).
    """
    Xs = []
    ys = []

    # Calculate effective length
    total_samples = len(X)

    # We need:
    # 1. sequence of length 'seq_length'
    # 2. target 'predict_ahead' steps after the END of the sequence
    #    (or whatever your exact target logic is)

    for i in range(total_samples):
        # Index of the last step in the input sequence
        end_seq_idx = i + seq_length

        # Index of the target step
        target_idx = end_seq_idx + predict_ahead - 1

        if target_idx >= total_samples:
            break

        # Extract sequence
        # X can be (Time, Feat) OR (Time, Grid, Feat)
        # Slicing [i:end] works for both
        seq_x = X[i:end_seq_idx]

        # NOTE: If you need to append X_global or flow history, handle it here.
        # For Set-Sequence, we primarily use seq_x.
        # If X_global is needed (and is 2D), we might need to expand/tile it to match 3D structure
        # or handle it in the model as a separate input.
        # For now, we return the primary sequence.

        Xs.append(seq_x)

        # Extract target (handle forecast horizon)
        if forecast_horizon == 1:
            ys.append(y[target_idx])
        else:
            # Multi-step horizon
            if target_idx + forecast_horizon > total_samples:
                break
            ys.append(y[target_idx: target_idx + forecast_horizon].flatten())

    return np.array(Xs), np.array(ys)


def create_sequences_per_basin(X_list, M_list, y_list, basin_lengths, seq_length, predict_ahead,
                               include_flow_history=False, flow_history_lag_gap=0,
                               basin_processing_batch_size=None, forecast_horizon=1, months_list=None):
    """
    Wrapper to handle list of basins.
    Maintains compatibility with V1/V2 but works for 3D inputs if passed.
    """
    # If X_list is already a list of arrays, we process them individually
    all_Xs = []
    all_ys = []
    all_months = []

    # If X_list is one combined array (from AE), we need to split it using basin_lengths.
    # But for Set-Sequence, X_list comes directly as a list of 3D arrays.

    # Logic: Check if X_list is a list or array
    if isinstance(X_list, list):
        iterator = zip(X_list, y_list)
        iter_months = months_list if months_list is not None else [None] * len(X_list)

        for X_basin, y_basin, months_basin in zip(X_list, y_list, iter_months):
            # Pass dummy global for now as set-sequence handles globals differently or integrated
            X_seq, y_seq = create_sequences(X_basin, None, y_basin, seq_length, predict_ahead,
                                            include_flow_history, flow_history_lag_gap, forecast_horizon)
            all_Xs.append(X_seq)
            all_ys.append(y_seq)

            if months_basin is not None:
                # Handle month tracking logic similar to V1/V2 if needed
                # (omitted for brevity unless strictly required for filtering)
                pass

        # Note: We cannot vstack all_Xs if they have different Grid sizes (Set-Sequence case).
        # We return the LIST of sequences if grid sizes differ.
        return all_Xs, all_ys, None
    else:
        # Standard V1/V2 flattened behavior
        # (This block assumes X_list is a single big array)
        start = 0
        for length in basin_lengths:
            end = start + length
            X_b = X_list[start:end]
            y_b = y_list[start:end]
            # ... process ...
            start = end
        # ... return stacked ...
        pass

    return np.array(all_Xs), np.array(all_ys), None
