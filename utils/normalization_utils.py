import numpy as np

DEFAULT_EPS = 1e-8


class StandardScalerNP:
    """A minimal NumPy standard scaler that works for 2D/3D/4D tensors.

    Fits mean/std on the *training* data only and can transform/inverse_transform
    along the last dimension (feature dimension).

    - For X shaped (..., n_features), it computes per-feature mean/std.
    - For y shaped (n_samples, 1) or (n_samples,), it behaves similarly.

    This avoids introducing sklearn as a hard dependency in the core pipeline
    logic and keeps shape-handling explicit.
    """

    def __init__(self, eps: float = DEFAULT_EPS):
        self.eps = float(eps)
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x)
        if x.size == 0:
            raise ValueError("Cannot fit scaler on empty array")

        # Collapse all non-feature axes.
        n_features = x.shape[-1] if x.ndim >= 1 else 1
        x2 = x.reshape(-1, n_features)
        self.mean_ = x2.mean(axis=0)
        self.std_ = x2.std(axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fit")
        x = np.asarray(x)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fit")
        x = np.asarray(x)
        return x * self.std_ + self.mean_


class RobustScalerNP:
    """
    Robust Scaler using Median and Interquartile Range (IQR).
    Robust to outliers.
    """

    def __init__(self, eps: float = DEFAULT_EPS, quantile_range=(25.0, 75.0)):
        self.eps = float(eps)
        self.center_ = None
        self.scale_ = None
        self.quantile_range = quantile_range

    def fit(self, x: np.ndarray):
        x = np.asarray(x)
        if x.size == 0:
            raise ValueError("Cannot fit scaler on empty array")

        # Collapse all non-feature axes.
        n_features = x.shape[-1] if x.ndim >= 1 else 1
        x2 = x.reshape(-1, n_features)

        q_min, q_max = self.quantile_range
        self.center_ = np.nanmedian(x2, axis=0)
        
        q25 = np.nanpercentile(x2, q_min, axis=0)
        q75 = np.nanpercentile(x2, q_max, axis=0)
        
        self.scale_ = q75 - q25
        # Avoid division by zero
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fit")
        x = np.asarray(x)
        return (x - self.center_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fit")
        x = np.asarray(x)
        return x * self.scale_ + self.center_


def fit_feature_scaler_from_basin_list(X_train_list):
    """Fit a StandardScalerNP on a list of basin tensors.

    Each basin tensor is expected to be shaped (Time, Grid, Feat).
    We concatenate over (Time*Grid) across basins.
    """
    if not isinstance(X_train_list, list) or len(X_train_list) == 0:
        raise ValueError("X_train_list must be a non-empty list")

    n_features = X_train_list[0].shape[-1]
    chunks = []
    for X in X_train_list:
        if X.ndim != 3:
            raise ValueError(f"Expected basin X to be 3D (Time, Grid, Feat), got {X.shape}")
        if X.shape[-1] != n_features:
            raise ValueError("All basins must share the same number of features")
        chunks.append(X.reshape(-1, n_features))

    agg = np.concatenate(chunks, axis=0)
    return StandardScalerNP().fit(agg)
