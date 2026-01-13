import numpy as np

def compute_basin_metrics(y_true, y_pred, epsilon=1e-6):
    """
    Compute comprehensive hydrological metrics.

    Returns:
        Dictionary with MSE, RMSE, MAE, R2, NSE, KGE
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Basic metrics
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # R2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + epsilon
    r2 = 1 - (ss_res / ss_tot)

    # NSE
    nse = 1 - (ss_res / ss_tot)

    # KGE
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    alpha = np.std(y_pred) / (np.std(y_true) + epsilon)
    beta = np.mean(y_pred) / (np.mean(y_true) + epsilon)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'NSE': float(nse),
        'KGE': float(kge)
    }
