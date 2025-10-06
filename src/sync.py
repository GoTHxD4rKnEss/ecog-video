# src/sync.py
# Robust synchronization utilities for aligning ECoG (1200 Hz) to video (30 Hz)
# - Edge detection with hysteresis + debounce
# - Offset estimation using event times (robust median)
# - Diagnostics + JSON save/load
# - Safe resampling helpers for 1200→30 Hz plots/correlations
# - Quick overlay plot for slides
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import json
import numpy as np
from scipy.signal import resample_poly


# --------------------------- Dataclasses ---------------------------

@dataclass
class OffsetResult:
    offset_seconds: float          # ecog time minus video time (s). Positive -> ECoG lags video
    offset_samples_ecog: int       # same offset in ECoG samples (fs_ecog)
    fs_ecog: float
    fps_video: float
    n_edges_ecog: int
    n_edges_video: int
    n_matched: int
    median_abs_error_ms: float     # median |time diff| for matched edges (ms)
    iqr_ms: float                  # interquartile range (ms)
    notes: str = ""

    def to_json(self) -> Dict:
        return asdict(self)


# --------------------------- Core helpers ---------------------------

def detect_edges_hysteresis(
    x: np.ndarray,
    low_th: float = 0.3,
    high_th: float = 0.7,
    min_isi_ms: float = 20.0,
    fs: float = 1200.0,
) -> np.ndarray:
    """
    Robust rising-edge detector for diode-like signals.
    - Hysteresis: must cross high_th to arm, then drop below low_th to re-arm
    - Debounce: require min inter-spike interval (ms)
    Returns: 1D array of edge indices
    """
    x = np.asarray(x, dtype=float)
    # Normalize to [0,1] robustly (guard against outliers)
    x_min, x_max = np.percentile(x, [1, 99])
    if x_max <= x_min:
        x_min, x_max = x.min(), x.max() if x.max() > x.min() else (0.0, 1.0)
    xn = np.clip((x - x_min) / (x_max - x_min + 1e-12), 0.0, 1.0)

    armed = True
    edges: List[int] = []
    min_isi = int(round((min_isi_ms / 1000.0) * fs))
    last_edge = -10**9

    for i in range(1, len(xn)):
        if armed and xn[i - 1] < high_th <= xn[i]:
            # rising through high threshold
            if i - last_edge >= min_isi:
                edges.append(i)
                last_edge = i
                armed = False
        elif not armed and xn[i] < low_th:
            # re-arm once we drop below low threshold
            armed = True

    return np.asarray(edges, dtype=int)


def _match_by_nearest(
    t_ref: np.ndarray, t_query: np.ndarray, max_diff_s: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each t_query event, find nearest t_ref event within max_diff_s.
    Returns: indices_ref, indices_query, diffs (t_query - t_ref) in seconds (only for matches)
    """
    if len(t_ref) == 0 or len(t_query) == 0:
        return np.array([], int), np.array([], int), np.array([], float)

    # two-pointer sweep (both sorted)
    i, j = 0, 0
    idx_ref, idx_qry, diffs = [], [], []
    while i < len(t_ref) and j < len(t_query):
        dt = t_query[j] - t_ref[i]
        # move the one that's behind
        if abs(dt) <= max_diff_s:
            idx_ref.append(i)
            idx_qry.append(j)
            diffs.append(dt)
            i += 1
            j += 1
        elif t_ref[i] < t_query[j]:
            i += 1
        else:
            j += 1

    return np.asarray(idx_ref), np.asarray(idx_qry), np.asarray(diffs)


def estimate_offset_from_edges(
    ecog_signal: np.ndarray,
    video_signal: np.ndarray,
    fs_ecog: float,
    fps_video: float,
    low_th: float = 0.3,
    high_th: float = 0.7,
    min_isi_ms: float = 20.0,
    max_match_diff_ms: float = 60.0,
    notes: str = "ECoG CH162 vs video_diode",
) -> OffsetResult:
    """
    Estimate ECoG↔video time offset using rising-edge event times.

    Steps:
    1) Detect edges in both signals with hysteresis + debounce
    2) Convert indices to seconds using each sampling rate
    3) Match nearest events within max_match_diff_ms
    4) Use the robust median of (t_ecog - t_video) as offset

    Returns OffsetResult with diagnostics.
    """
    # 1) edges
    edges_ecog = detect_edges_hysteresis(ecog_signal, low_th, high_th, min_isi_ms, fs=fs_ecog)
    edges_vid  = detect_edges_hysteresis(video_signal, low_th, high_th, min_isi_ms, fs=fps_video)

    t_ecog = edges_ecog / fs_ecog
    t_vid  = edges_vid  / fps_video

    # 2) match
    idx_e, idx_v, diffs = _match_by_nearest(t_ref=t_vid, t_query=t_ecog, max_diff_s=max_match_diff_ms/1000.0)

    if diffs.size == 0:
        raise RuntimeError(
            "No matched diode events found within the allowed window. "
            "Check thresholds or increase max_match_diff_ms."
        )

    # ecog_time - video_time (seconds)
    offset_s = np.median(diffs)          # robust to outliers
    mad_s = np.median(np.abs(diffs - offset_s)) + 1e-12
    iqr_ms = float(np.percentile(diffs, 75) - np.percentile(diffs, 25)) * 1000.0

    offset_samples = int(round(offset_s * fs_ecog))

    return OffsetResult(
        offset_seconds=float(offset_s),
        offset_samples_ecog=offset_samples,
        fs_ecog=float(fs_ecog),
        fps_video=float(fps_video),
        n_edges_ecog=int(len(edges_ecog)),
        n_edges_video=int(len(edges_vid)),
        n_matched=int(len(diffs)),
        median_abs_error_ms=float(np.median(np.abs(diffs - offset_s)) * 1000.0),
        iqr_ms=iqr_ms,
        notes=notes,
    )


def save_offset(result: OffsetResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_json(), f, indent=2)
    print(f"[sync] Saved offset → {out_path}")


def load_offset(path: Path) -> OffsetResult:
    d = json.load(open(Path(path), "r"))
    return OffsetResult(**d)


# --------------------------- Resampling helpers ---------------------------

def resample_to_rate(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample 1D or 2D time series to a new sampling rate using polyphase filtering.
    x shape: (time,) or (time, channels)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    # rational approximation
    from math import gcd
    up = int(round(fs_out * 1000))
    down = int(round(fs_in * 1000))
    g = gcd(up, down)
    up //= g; down //= g
    y = resample_poly(x, up, down, axis=0)
    return y.squeeze()


def shift_samples(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift a 1D or (time, channels) array by 'shift' samples (positive shifts right).
    Pad with edge values to keep length.
    """
    x = np.asarray(x)
    if shift == 0:
        return x
    if x.ndim == 1:
        y = np.empty_like(x)
        if shift > 0:
            y[:shift] = x[0]
            y[shift:] = x[:-shift]
        else:
            s = -shift
            y[-s:] = x[-1]
            y[:-s] = x[s:]
        return y
    else:
        y = np.empty_like(x)
        if shift > 0:
            y[:shift, :] = x[0, :]
            y[shift:, :] = x[:-shift, :]
        else:
            s = -shift
            y[-s:, :] = x[-1, :]
            y[:-s, :] = x[s:, :]
        return y


# --------------------------- Plot helper (optional) ---------------------------

def plot_diode_overlay(
    ecog_diode: np.ndarray,
    video_diode: np.ndarray,
    offset_samples_ecog: int,
    fs_ecog: float,
    fps_video: float,
    seconds: float = 3.0,
    out_path: Optional[Path] = None,
):
    """
    Quick sanity plot: overlay normalized video diode (upsampled to fs_ecog)
    and ECoG diode shifted by the estimated offset.
    """
    import matplotlib.pyplot as plt

    n_ecog = int(seconds * fs_ecog)
    n_vid  = int(seconds * fps_video)

    # Shift ECoG diode, resample video diode to ECoG rate for overlay
    ecog_shifted = shift_samples(ecog_diode, offset_samples_ecog)
    video_up = resample_to_rate(video_diode[:n_vid], fps_video, fs_ecog)

    t = np.arange(len(video_up)) / fs_ecog
    e = ecog_shifted[:len(video_up)]

    # z-norm for overlay
    z = lambda a: (a - a.mean()) / (a.std() + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(t, z(video_up), label="Video diode ↑ to 1200 Hz", lw=1.5)
    plt.plot(t, z(e), label="ECoG diode (shifted)", lw=1.2)
    plt.xlabel("Time (s)"); plt.ylabel("z-score")
    plt.title(f"Diode alignment (first {seconds:.1f}s) | shift={offset_samples_ecog} samples")
    plt.legend(); plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"[sync] Saved overlay → {out_path}")
    plt.show()
