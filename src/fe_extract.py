# src/fe_extract.py
# -------------------------------------------------------------
# Trial-based feature extraction for ECoG (beginner-friendly).
# Inputs (in a folder, typically data/derivatives/):
#   - epochs.npy        : (trials, channels, samples) OR (trials, samples, channels)
#   - time_vector.npy   : (samples,) time in milliseconds [-300 .. +400]
#   - stimcode.npy      : (trials,) integer labels (optional but recommended)
#   - good_channels.npy : (n_good,) indices of good channels (recommended)
#
# Outputs:
#   - Broadband gamma features (110–140 Hz), log-variance in 100–400 ms
#   - Optional canonical band powers (theta/alpha/beta/gamma) in 100–400 ms
#   - Saved to data/derivatives/trials_features.npz + feature_summary.csv
# -------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch, hilbert


# ------------------ parameters we’ll reuse ------------------

@dataclass
class FeatureParams:
    # Windows in milliseconds relative to stimulus onset
    baseline_ms: Tuple[float, float] = (-300.0, 0.0)    # “normal” period, before onset
    feat_ms: Tuple[float, float] = (100.0, 400.0)       # where the response should be strongest
    # Main feature band (broadband gamma)
    gamma_band: Tuple[float, float] = (110.0, 140.0)
    # Comparison bands
    canonical_bands: Tuple[Tuple[float, float], ...] = ((4, 8), (8, 13), (13, 30), (30, 80))
    # Filter design
    filt_order: int = 4
    # Tell us if epochs were already baseline-normalized
    # (If False, we’ll z-score using baseline_ms.)
    epochs_are_zscored: bool = True
    # If your team used % change baseline (per preprocessing_params), you can compute % change too.
    use_percent_change: bool = False


# ------------------ small helpers ------------------

def _indices_from_ms(time_ms: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    """Boolean mask for samples between start_ms and end_ms (inclusive)."""
    return (time_ms >= start_ms) & (time_ms <= end_ms)

def _estimate_fs_from_time_ms(time_ms: np.ndarray) -> float:
    """Compute sampling rate (Hz) from ms spacing (should be ~1200 Hz)."""
    step_ms = float(np.median(np.diff(time_ms)))  # robust to tiny jitter
    if step_ms <= 0:
        raise ValueError("time_vector spacing must be positive.")
    return 1000.0 / step_ms

def load_inputs(base_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load epochs, time vector, labels, and good channels from a directory.
    Reorders epochs to (trials, samples, channels) if needed.
    """
    base_dir = Path(base_dir)
    epochs_p = base_dir / "epochs.npy"
    time_p   = base_dir / "time_vector.npy"
    stim_p   = base_dir / "stimcode.npy"
    good_p   = base_dir / "good_channels.npy"

    if not epochs_p.exists():
        raise FileNotFoundError(f"Missing {epochs_p}. Place your npy files in {base_dir}")

    epochs = np.load(epochs_p)                 # could be (T, C, S) or (T, S, C)
    time_ms = np.load(time_p) if time_p.exists() else None
    labels = np.load(stim_p) if stim_p.exists() else None
    good = np.load(good_p) if good_p.exists() else None

    # Reorder to (T, S, C) if time_vector exists
    epochs = np.asarray(epochs)
    if time_ms is not None:
        S = len(time_ms)
        if epochs.shape[-1] == S and epochs.shape[1] != S:
            # epochs is (T, C, S) → make it (T, S, C)
            epochs = np.transpose(epochs, (0, 2, 1))
        elif epochs.shape[1] == S:
            pass  # already (T, S, C)
        else:
            raise ValueError(f"Epoch dims {epochs.shape} do not match time_vector length {S}")
    else:
        # If time_ms missing, we assume (T, S, C)
        pass

    return {"epochs": epochs, "time_ms": time_ms, "labels": labels, "good_ch": good}

def apply_good_channels(X: np.ndarray, good_ch: Optional[np.ndarray]) -> np.ndarray:
    """
    Keep only channels listed in good_ch. X must be (T, S, C).
    Accepts 0-based or 1-based indices and cleans them safely.
    """
    if good_ch is None:
        return X

    C = X.shape[2]
    idx = np.asarray(good_ch)

    # Convert boolean mask → integer indices
    if idx.dtype == bool:
        idx = np.where(idx)[0]
    else:
        idx = idx.astype(int)

    # Auto-detect 1-based (common in electrode lists)
    if idx.size > 0 and idx.min() >= 1 and idx.max() <= C:
        idx = idx - 1

    # De-duplicate & bounds-check
    idx = np.unique(idx)
    idx = idx[(idx >= 0) & (idx < C)]

    if idx.size == 0:
        raise ValueError(
            "After normalization, no valid good channels remain. "
            "Check good_channels.npy against epochs channel count."
        )

    return X[:, :, idx]



def baseline_zscore(X: np.ndarray, time_ms: np.ndarray, baseline_ms: Tuple[float, float]) -> np.ndarray:
    """
    Per-trial, per-channel z-score:
      z = (value - baseline_mean) / baseline_std
    This makes all trials/channels comparable.
    """
    mask = _indices_from_ms(time_ms, baseline_ms[0], baseline_ms[1])
    base = X[:, mask, :]                # (T, Sb, C)
    mu = base.mean(axis=1, keepdims=True)
    sd = base.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd

def baseline_percent_change(X: np.ndarray, time_ms: np.ndarray, baseline_ms: Tuple[float, float]) -> np.ndarray:
    """
    Per-trial, per-channel % change:
      pct = 100 * (value - baseline_mean) / (baseline_mean + eps)
    Useful if your preprocessing policy prefers percentage change over z-score.
    """
    mask = _indices_from_ms(time_ms, baseline_ms[0], baseline_ms[1])
    base = X[:, mask, :]
    mu = base.mean(axis=1, keepdims=True)
    return 100.0 * (X - mu) / (np.abs(mu) + 1e-8)

def bandpass_filtfilt(X: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase band-pass filter (no phase lag) along the time axis.
    X: (trials, samples, channels) → returns same shape.
    """
    ny = fs / 2.0
    b, a = butter(order, [lo/ny, hi/ny], btype="band")
    T, S, C = X.shape
    Xf = np.empty_like(X)
    for ch in range(C):
        Xf[:, :, ch] = filtfilt(b, a, X[:, :, ch], axis=1)
    return Xf

def logvar_in_window(X: np.ndarray, time_ms: np.ndarray, win_ms: Tuple[float, float]) -> np.ndarray:
    """
    Compute log-variance (≈ log power) per trial/channel inside a time window.
    Returns shape: (trials, channels).
    """
    mask = _indices_from_ms(time_ms, win_ms[0], win_ms[1])
    seg = X[:, mask, :]                 # (T, Swin, C)
    var = (seg ** 2).mean(axis=1)       # (T, C)
    return np.log(var + 1e-8)


# ------------------ Feature A: Broadband gamma ------------------

def features_broadband_gamma(
    epochs: np.ndarray, time_ms: np.ndarray, params: FeatureParams
) -> np.ndarray:
    """
    Steps:
      1) If epochs aren’t already baseline-normalized, do z-score or % change.
      2) Band-pass 110–140 Hz.
      3) Compute log-variance in 100–400 ms.
    Output: (trials, channels)
    """
    fs = _estimate_fs_from_time_ms(time_ms)
    X = epochs.copy()

    if not params.epochs_are_zscored:
        X = baseline_percent_change(X, time_ms, params.baseline_ms) if params.use_percent_change \
            else baseline_zscore(X, time_ms, params.baseline_ms)

    Xg = bandpass_filtfilt(X, fs, params.gamma_band[0], params.gamma_band[1], order=params.filt_order)
    feats = logvar_in_window(Xg, time_ms, params.feat_ms)
    return feats


# ------------------ Feature B: Canonical bands (optional) ------------------

def features_canonical_bands(
    epochs: np.ndarray, time_ms: np.ndarray, params: FeatureParams, method: str = "welch"
) -> np.ndarray:
    """
    Compute per-trial/channel power in standard bands (theta/alpha/beta/gamma) over 100–400 ms.
    Two methods:
      - 'welch'  : compute PSD then average within each band (robust & simple)
      - 'hilbert': filter each band, take envelope (Hilbert), square and average
    Output: (trials, channels * n_bands)
    """
    fs = _estimate_fs_from_time_ms(time_ms)
    X = epochs.copy()

    if not params.epochs_are_zscored:
        X = baseline_percent_change(X, time_ms, params.baseline_ms) if params.use_percent_change \
            else baseline_zscore(X, time_ms, params.baseline_ms)

    # cut the feature window (100–400 ms)
    mask = _indices_from_ms(time_ms, params.feat_ms[0], params.feat_ms[1])
    seg = X[:, mask, :]                 # (T, Swin, C)
    T, Sw, C = seg.shape

    blocks = []
    if method == "welch":
        for lo, hi in params.canonical_bands:
            band_pow = np.zeros((T, C), dtype=float)
            for t in range(T):
                f, Pxx = welch(seg[t], fs=fs, axis=0, nperseg=min(256, Sw))
                m = (f >= lo) & (f <= hi)
                band_pow[t] = Pxx[m, :].mean(axis=0) if m.any() else 0.0
            blocks.append(np.log(band_pow + 1e-12))
    else:
        for lo, hi in params.canonical_bands:
            Xb = bandpass_filtfilt(seg, fs, lo, hi, order=params.filt_order)
            env = np.abs(hilbert(Xb, axis=1))
            pow_mean = (env ** 2).mean(axis=1)  # (T, C)
            blocks.append(np.log(pow_mean + 1e-8))

    return np.concatenate(blocks, axis=1)   # (T, C * n_bands)


# ------------------ Saving ------------------

def save_features(
    out_dir: Path, feats_gamma: np.ndarray, labels: Optional[np.ndarray], params: FeatureParams,
    band_feats: Optional[np.ndarray] = None
) -> Path:
    """
    Save to .npz with metadata so modelers know the windows/bands used.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = asdict(params)
    outp = out_dir / "trials_features.npz"
    if labels is None:
        np.savez(outp, gamma=feats_gamma, bands=band_feats, meta=meta)
    else:
        np.savez(outp, gamma=feats_gamma, bands=band_feats, labels=labels, meta=meta)
    return outp
