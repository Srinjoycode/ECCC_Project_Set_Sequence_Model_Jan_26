from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _aggregate_overlapping_predictions_mean(y_pred_2d: np.ndarray) -> np.ndarray:
    """Same aggregation used in evaluation: mean over all overlapping horizon predictions."""
    y_pred_2d = np.asarray(y_pred_2d)
    if y_pred_2d.ndim != 2:
        raise ValueError(f"Expected (N,H), got {y_pred_2d.shape}")

    n, h = y_pred_2d.shape
    out_len = n + h - 1
    sums = np.zeros(out_len, dtype=np.float64)
    counts = np.zeros(out_len, dtype=np.int32)

    for i in range(n):
        for k in range(h):
            t = i + k
            sums[t] += float(y_pred_2d[i, k])
            counts[t] += 1

    counts = np.maximum(counts, 1)
    return (sums / counts).astype(np.float64)


def _aggregate_to_h1_timeline_mean(y_pred_2d: np.ndarray) -> np.ndarray:
    """Aggregate multi-horizon predictions onto the H1 timeline.

    For y_pred_2d shape (N,H), returns y_mean shape (N,) where index j aggregates
    all predictions y_pred[i,k] that map to H1 index j=i+k (with j < N).
    """
    y_pred_2d = np.asarray(y_pred_2d)
    if y_pred_2d.ndim != 2:
        raise ValueError(f"Expected (N,H), got {y_pred_2d.shape}")

    n, h = y_pred_2d.shape
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        for k in range(h):
            j = i + k
            if 0 <= j < n:
                sums[j] += float(y_pred_2d[i, k])
                counts[j] += 1

    counts = np.maximum(counts, 1)
    return (sums / counts).astype(np.float64)


def _plot_single_series(y_true_1d, y_pred_1d, phase_name, output_dir, filename, title_suffix="", filter_months=None):
    y_true_1d = np.asarray(y_true_1d).reshape(-1)
    y_pred_1d = np.asarray(y_pred_1d).reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(y_true_1d, label='Observed', color='blue', alpha=0.7, linewidth=1.5)
    ax.plot(y_pred_1d, label='Predicted', color='red', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Flow (m³/s)')
    title = f'{phase_name} - Time Series{title_suffix}'
    if filter_months:
        title += f' (Months: {filter_months})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(y_true_1d, y_pred_1d, alpha=0.5, s=20)
    min_val = float(min(y_true_1d.min(), y_pred_1d.min()))
    max_val = float(max(y_true_1d.max(), y_pred_1d.max()))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Observed Flow (m³/s)')
    ax.set_ylabel('Predicted Flow (m³/s)')
    ax.set_title(f'{phase_name} - Scatter Plot{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    r2 = r2_score(y_true_1d, y_pred_1d) if len(y_true_1d) > 1 else float('nan')
    ax.text(
        0.05, 0.95,
        f'R² = {r2:.4f}' if np.isfinite(r2) else 'R² = n/a',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def plot_predictions(
        y_true,
        y_pred,
        phase_name,
        output_dir,
        filter_months=None,
        target_months_seq=None,
        restrict_mean_to_filter_months=False,
):
    """Plot predictions.

    - If single-step: writes one file: predictions_<phase>.png (legacy behavior)
    - If multi-horizon (N,H): writes:
        * predictions_<phase>_h1.png ... _hH.png (one plot per horizon)
        * predictions_<phase>_mean.png (aggregated-mean overlapping forecast)
    """
    y_true_2d = _ensure_2d(y_true)
    y_pred_2d = _ensure_2d(y_pred)

    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_true_2d.shape} vs {y_pred_2d.shape}")

    # Legacy single-step path
    if y_true_2d.shape[1] == 1:
        filename = f'predictions_{phase_name.replace(" ", "_")}.png'
        _plot_single_series(
            y_true_2d[:, 0],
            y_pred_2d[:, 0],
            phase_name,
            output_dir,
            filename,
            title_suffix="",
            filter_months=filter_months,
        )
        return

    # Multi-horizon: one plot per horizon
    h = y_true_2d.shape[1]
    for k in range(h):
        step = k + 1
        filename = f'predictions_{phase_name.replace(" ", "_")}_h{step}.png'
        _plot_single_series(
            y_true_2d[:, k],
            y_pred_2d[:, k],
            phase_name,
            output_dir,
            filename,
            title_suffix=f" (Horizon {step})",
            filter_months=filter_months,
        )

    # H1-anchored mean plot (preferred)
    y_true_h1 = y_true_2d[:, 0]
    y_pred_mean_h1 = _aggregate_to_h1_timeline_mean(y_pred_2d)

    if restrict_mean_to_filter_months and filter_months is not None and target_months_seq is not None:
        tm = np.asarray(target_months_seq).reshape(-1)
        n = min(len(tm), len(y_true_h1))
        if n > 0:
            mask = np.isin(tm[:n].astype(np.int32), np.asarray(filter_months, dtype=np.int32))
            y_true_h1 = y_true_h1[:n][mask]
            y_pred_mean_h1 = y_pred_mean_h1[:n][mask]

    filename = f'predictions_{phase_name.replace(" ", "_")}_mean.png'
    _plot_single_series(
        y_true_h1,
        y_pred_mean_h1,
        phase_name,
        output_dir,
        filename,
        title_suffix=" (H1-Anchored Mean)",
        filter_months=filter_months,
    )

    # Optional legacy (N+H-1) aggregated plot for comparison
    y_true_mean_legacy = _aggregate_overlapping_predictions_mean(y_true_2d)
    y_pred_mean_legacy = _aggregate_overlapping_predictions_mean(y_pred_2d)
    filename = f'predictions_{phase_name.replace(" ", "_")}_mean_legacy.png'
    _plot_single_series(
        y_true_mean_legacy,
        y_pred_mean_legacy,
        phase_name,
        output_dir,
        filename,
        title_suffix=" (Aggregated Mean - Legacy)",
        filter_months=filter_months,
    )


def plot_training_history(history, phase_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{phase_name} - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if 'mae' in history.history:
        ax.plot(history.history['mae'], label='Training MAE')
        ax.plot(history.history['val_mae'], label='Validation MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title(f'{phase_name} - MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'training_history_{phase_name.replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history: {filename}")


def plot_multistage_training(histories, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    colors = ['blue', 'green', 'orange', 'red', 'purple']
    stage_names = [
        'Stage 1\n(Global GRU)',
        'Stage 2\n(Attention+MLP)',
        'Stage 3\n(Global+Attn+MLP)',
        'Stage 4\n(+Top LSTM)',
        'Stage 5\n(All Layers)'
    ]

    epoch_offset = 0
    ax = axes[0]
    for i, (stage, history) in enumerate(histories.items()):
        epochs = range(epoch_offset, epoch_offset + len(history.history['loss']))
        ax.plot(epochs, history.history['loss'],
                color=colors[i], alpha=0.7, linewidth=2,
                label=f'{stage_names[i]} (Train)')
        ax.plot(epochs, history.history['val_loss'],
                color=colors[i], alpha=0.7, linewidth=2,
                linestyle='--', label=f'{stage_names[i]} (Val)')
        if i < len(histories) - 1:
            ax.axvline(x=epoch_offset + len(history.history['loss']),
                       color='gray', linestyle=':', alpha=0.5)
        epoch_offset += len(history.history['loss'])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Multi-Stage Training: Loss')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1]
    epoch_offset = 0
    for i, (stage, history) in enumerate(histories.items()):
        epochs = range(epoch_offset, epoch_offset + len(history.history['mae']))
        ax.plot(epochs, history.history['mae'],
                color=colors[i], alpha=0.7, linewidth=2,
                label=f'{stage_names[i]} (Train)')
        ax.plot(epochs, history.history['val_mae'],
                color=colors[i], alpha=0.7, linewidth=2,
                linestyle='--', label=f'{stage_names[i]} (Val)')
        if i < len(histories) - 1:
            ax.axvline(x=epoch_offset + len(history.history['mae']),
                       color='gray', linestyle=':', alpha=0.5)
        epoch_offset += len(history.history['mae'])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Multi-Stage Training: MAE')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multistage_training_progress.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved multi-stage training plot: multistage_training_progress.png")
