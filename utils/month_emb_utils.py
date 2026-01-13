import numpy as np


def get_month_dim(config):
    if config.MONTH_ENCODING == 'sinusoidal':
        return config.MONTH_EMB_DIM
    return 12


def month_sinusoidal_embedding(months, dim=12, period=12):
    months = np.asarray(months).astype(int).reshape(-1) - 1  # 0..11
    if dim % 2 != 0:
        raise ValueError('MONTH_EMB_DIM must be even for sin/cos pairs')
    positions = months[:, None].astype(np.float32)
    d = dim
    half = d // 2
    i = np.arange(half)[None, :]
    angle_rates = 1.0 / (10000 ** ((2.0 * i) / d))
    base = (2.0 * np.pi / period)
    angles = positions * base * angle_rates
    s = np.sin(angles)
    c = np.cos(angles)
    emb = np.empty((months.shape[0], d), dtype=np.float32)
    emb[:, 0::2] = s
    emb[:, 1::2] = c
    return emb