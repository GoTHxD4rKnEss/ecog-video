"""
config.py
----------
Central place for all constants and small helpers used across the project.

Tip:
- Import from notebooks and modules like:
    from src import config as C
- Change values here (e.g., offset, bands), and the whole repo stays consistent.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import numpy as np


# ========== Project paths ==========
# Resolve repository root by walking up until we see 'src' and 'data'
def find_repo_root(start: Optional[Path] = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for _ in range(8):
        if (p / "src").exists() and (p / "data").exists():
            return p
        p = p.parent
    return (start or Path.cwd()).resolve()

REPO_ROOT: Path = find_repo_root()
DATA_DIR: Path = REPO_ROOT / "data"
DERIV_DIR: Path = DATA_DIR / "derivatives"
RAW_DIR: Path = DATA_DIR / "raw"
EXTERNAL_DIR: Path = DATA_DIR / "external"
FIG_DIR: Path = (REPO_ROOT / "reports" / "figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ========== Sampling + basic info ==========
FS: float = 1200.0          # ECoG sampling rate (Hz)
FPS_VIDEO: float = 30.0     # Nominal video frame rate; we often infer actual FPS from data
DURATION_S: float = 252.0   # Movie duration (seconds), from Walk_paradigm.xml

# ========== Channel layout (0-based indices) ==========
# Continuous .npy (filtered_data.npy) provided by teammate is (channels, samples)
# ECoG channels: 160 channels; CH1..CH160  -> indices 0..159
# Photodiode (CH162), StimCode (CH163), GroupId (CH164) are NOT in filtered_data.npy,
# but they exist in the original .mat. We keep their conventional indices here for reference.
ECOG_N_CHANNELS: int = 160
PHOTODIODE_CH_0B: int = 161     # CH162 -> index 161 (if present in an array)
STIMCODE_CH_0B: int   = 162     # CH163 -> index 162 (if present)
GROUPID_CH_0B: int    = 163     # CH164 -> index 163 (if present)

# ========== Alignment (video <-> ECoG) ==========
# We store the measured offset (in ECoG samples) in alignment_offset.json.
# Positive offset means: ecog_time - video_time > 0  (ECoG lags video).
# Negative offset means: ECoG leads video and must be shifted backward.
ALIGNMENT_JSON: Path = DERIV_DIR / "alignment_offset.json"

def load_alignment_offset(default_samples: int = -425) -> int:
    """
    Load the alignment offset (in ECoG samples) from data/derivatives/alignment_offset.json.
    Falls back to `default_samples` if the file is missing or malformed.
    """
    try:
        d = json.load(open(ALIGNMENT_JSON, "r", encoding="utf-8"))
        # prefer explicit key if present
        if "offset_samples_ecog" in d:
            return int(d["offset_samples_ecog"])
        # legacy/alternate key
        if "offset_samples" in d:
            return int(d["offset_samples"])
    except Exception:
        pass
    return int(default_samples)

# Expose as a constant for convenience
ALIGNMENT_OFFSET_SAMPLES: int = load_alignment_offset(default_samples=-425)


# ========== Feature definitions ==========
# Main feature: Broadband high-gamma (HG)
HG_BAND: Tuple[float, float] = (110.0, 140.0)         # Hz
HG_SMOOTH_MS: float = 25.0                            # optional moving-average smoothing of Hilbert amp

# Canonical bands (optional)
BANDS_CANONICAL: Tuple[Tuple[float, float], ...] = (
    (4.0, 7.0),    # theta
    (8.0, 12.0),   # alpha
    (13.0, 30.0),  # beta
    (30.0, 70.0),  # low-gamma
)

# Time windows relative to VIDEO time (after applying offset)
BASELINE_MS: Tuple[float, float] = (-300.0, 0.0)
RESPONSE_MS: Tuple[float, float] = (100.0, 400.0)

# Welch PSD params (for canonical bands)
WELCH_WIN_MS: float = 256.0
WELCH_OVERLAP: float = 0.5     # 50%
WELCH_NFFT: Optional[int] = None  # let scipy choose based on window length


# ========== File names we standardize around ==========
FILTERED_CONTINUOUS_NPY: Path = DERIV_DIR / "filtered_data.npy"       # (channels, samples)
VIDEO_DIODE_NPY: Path        = DERIV_DIR / "video_diode.npy"          # (frames,) ~30 Hz
TRIAL_ONSETS_NPY: Path       = DERIV_DIR / "trial_onsets.npy"         # (n_trials,) ecog sample indices
GOOD_CHANNELS_NPY: Path      = DERIV_DIR / "good_channels.npy"        # indices (0- or 1-based handled by helper)
EPOCHS_NPY: Path             = DERIV_DIR / "epochs.npy"               # (trials, samples, channels) or (trials, channels, samples)
TIME_VECTOR_NPY: Path        = DERIV_DIR / "time_vector.npy"          # (samples,) ms
STIMCODE_NPY: Path           = DERIV_DIR / "stimcode.npy"             # may be constant (block id)
TRIAL_LABELS_NPY: Path       = DERIV_DIR / "trial_labels.npy"         # (n_trials,) optional true labels

# Outputs of (re)feature extraction with alignment applied
CONT_HG_ENV_NPZ: Path        = DERIV_DIR / "continuous_hg_env.npz"    # env (samples x channels), fs, offset, channels_used
TRIALS_FEATS_NPZ: Path       = DERIV_DIR / "trials_features_aligned.npz"
FEAT_SUMMARY_CSV: Path       = DERIV_DIR / "feature_summary_aligned.csv"


# ========== Small helpers (safe to import anywhere) ==========

def load_good_channels(path: Path = GOOD_CHANNELS_NPY, n_channels: int = ECOG_N_CHANNELS) -> np.ndarray:
    """
    Load channel indices and normalize to 0-based integer indices.
    Accepts:
      - 0-based indices (min >= 0)
      - 1-based indices (min >= 1) -> will be converted to 0-based
      - boolean masks of length n_channels
    """
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.dtype == bool:
        if arr.size != n_channels:
            raise ValueError(f"Boolean good_channels has length {arr.size}, expected {n_channels}.")
        idx = np.where(arr)[0]
        return idx.astype(int)

    # integer-like array
    idx = arr.astype(int).ravel()
    if idx.min() >= 1 and idx.max() <= n_channels:
        # looks like 1-based -> convert
        idx = idx - 1
    # basic bounds check
    if idx.min() < 0 or idx.max() >= n_channels:
        raise ValueError(f"good_channels indices out of bounds for n_channels={n_channels}: min={idx.min()}, max={idx.max()}")
    return idx.astype(int)


@dataclass(frozen=True)
class Names:
    """Canonical figure names to keep slides reproducible."""
    timeline: str = "01_timeline.png"
    raw_snippets: str = "02_raw_snippets.png"
    psd_compare: str = "03_psd_video_vs_pre.png"
    bad_variance: str = "04_bad_channel_variance.png"
    spectrogram: str = "05_quick_spectrogram.png"
    align_overlay: str = "align_01_diode_overlay.png"
    align_phase_overlay: str = "align_phase_overlay.png"
    impulse_overlay: str = "align_02_impulse_overlay.png"
    feat_pca: str = "fe_03_pca2_scatter.png"
    feat_kmeans: str = "fe_04_kmeans_clusters.png"
    channel_corr: str = "fe_05_channel_corr.png"

FIGURES = Names()


def describe() -> Dict[str, object]:
    """
    Quick snapshot of key configuration for logging in notebooks.
    """
    return {
        "repo_root": str(REPO_ROOT),
        "fs_ecog": FS,
        "fps_video_nominal": FPS_VIDEO,
        "alignment_offset_samples": ALIGNMENT_OFFSET_SAMPLES,
        "hg_band": HG_BAND,
        "baseline_ms": BASELINE_MS,
        "response_ms": RESPONSE_MS,
        "welch_window_ms": WELCH_WIN_MS,
        "derivatives_dir": str(DERIV_DIR),
    }
