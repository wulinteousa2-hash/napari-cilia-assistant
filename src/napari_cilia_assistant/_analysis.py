from __future__ import annotations

"""Core numerical routines for napari-cilia-assistant.

This module keeps the scientific computation separate from the Qt/napari UI.
The routines are intentionally transparent and conservative: they expose raw
signals, spectra, maps, and quality metrics so users can review the measurement
rather than relying on a single black-box number.
"""

from contextlib import suppress

import cv2
import numpy as np
from scipy.signal import detrend, find_peaks, periodogram, welch


ROI = tuple[int, int, int, int]  # x, y, width, height


def read_avi_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        codec = codec if codec.strip("\x00 ") else "unknown"

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": frame_count / fps if fps > 0 else np.nan,
            "width": width,
            "height": height,
            "codec": codec,
        }
    finally:
        with suppress(Exception):
            cap.release()


def load_avi_as_stack(path: str, max_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        frames: list[np.ndarray] = []
        index = 0
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray)
            index += 1
    finally:
        with suppress(Exception):
            cap.release()

    if not frames:
        raise ValueError("No frames were loaded from the AVI file.")

    stack = np.stack(frames, axis=0)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape T,Y,X, got {stack.shape}")
    return stack


def summarize_stack(stack: np.ndarray) -> dict:
    return {
        "shape": tuple(stack.shape),
        "dtype": str(stack.dtype),
        "min": float(np.min(stack)),
        "max": float(np.max(stack)),
        "mean": float(np.mean(stack)),
    }


def roi_from_shape_data(shape_data: np.ndarray, image_shape: tuple[int, int]) -> ROI:
    data = np.asarray(shape_data)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid shape data: {data.shape}")

    y_coords = data[:, -2]
    x_coords = data[:, -1]

    y0 = int(np.floor(np.min(y_coords)))
    y1 = int(np.ceil(np.max(y_coords)))
    x0 = int(np.floor(np.min(x_coords)))
    x1 = int(np.ceil(np.max(x_coords)))

    img_y, img_x = image_shape
    y0 = max(0, min(y0, img_y - 1))
    y1 = max(1, min(y1, img_y))
    x0 = max(0, min(x0, img_x - 1))
    x1 = max(1, min(x1, img_x))

    width = x1 - x0
    height = y1 - y0
    if width <= 1 or height <= 1:
        raise ValueError("ROI is too small. Draw a larger rectangle over the moving cilia.")
    return x0, y0, width, height


def crop_stack_to_roi(stack: np.ndarray, roi: ROI | None = None) -> np.ndarray:
    if roi is None:
        return stack
    x, y, w, h = roi
    return stack[:, y : y + h, x : x + w]


def roi_mean_signal(stack: np.ndarray, roi: ROI | None = None) -> np.ndarray:
    cropped = crop_stack_to_roi(stack, roi)
    return cropped.mean(axis=(1, 2)).astype(float)


def _effective_frequency_range(fps: float, min_hz: float, max_hz: float) -> tuple[float, float, float]:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    nyquist_hz = fps / 2.0
    effective_max_hz = min(max_hz, nyquist_hz * 0.95)
    if min_hz >= effective_max_hz:
        raise ValueError(
            f"Invalid frequency range. With FPS={fps:.3f}, max useful frequency is ~{nyquist_hz:.3f} Hz."
        )
    return min_hz, effective_max_hz, nyquist_hz


def _prepare_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("Expected a one-dimensional signal.")
    if signal.size < 16:
        raise ValueError("Signal is too short for frequency analysis.")
    clean = detrend(signal)
    clean = clean - np.mean(clean)
    if float(np.std(clean)) <= 1e-12:
        raise ValueError("Signal has no measurable intensity variation.")
    return clean


def estimate_cbf_frequency(
    signal: np.ndarray,
    fps: float,
    min_hz: float = 3.0,
    max_hz: float = 40.0,
    method: str = "fft",
) -> dict:
    """Estimate CBF from a 1-D time signal using FFT, periodogram, or Welch PSD."""
    min_hz, effective_max_hz, nyquist_hz = _effective_frequency_range(fps, min_hz, max_hz)
    clean = _prepare_signal(signal)
    method_key = method.strip().lower()

    if method_key == "fft":
        windowed = clean * np.hanning(clean.size)
        freqs = np.fft.rfftfreq(windowed.size, d=1.0 / fps)
        power = np.abs(np.fft.rfft(windowed)) ** 2
        method_name = "FFT dominant frequency"
    elif method_key == "periodogram":
        freqs, power = periodogram(clean, fs=fps, window="hann", detrend=False, scaling="spectrum")
        method_name = "Periodogram dominant frequency"
    elif method_key == "welch":
        nperseg = min(clean.size, max(16, int(round(fps))))
        freqs, power = welch(clean, fs=fps, window="hann", nperseg=nperseg, detrend=False, scaling="spectrum")
        method_name = "Welch PSD dominant frequency"
    else:
        raise ValueError(f"Unsupported frequency method: {method}. Use FFT, Welch, or Periodogram.")

    valid = (freqs >= min_hz) & (freqs <= effective_max_hz)
    if not np.any(valid):
        raise ValueError("No valid frequency bins in selected range.")

    valid_freqs = freqs[valid]
    valid_power = power[valid]
    peak_index = int(np.argmax(valid_power))
    cbf_hz = float(valid_freqs[peak_index])
    peak_power = float(valid_power[peak_index])
    background_power = float(np.median(valid_power) + 1e-12)

    return {
        "method": method_name,
        "cbf_hz": cbf_hz,
        "peak_to_background": peak_power / background_power,
        "freqs": freqs,
        "power": power,
        "effective_max_hz": effective_max_hz,
        "nyquist_hz": nyquist_hz,
    }


def estimate_cbf_fft(signal: np.ndarray, fps: float, min_hz: float = 3.0, max_hz: float = 40.0) -> dict:
    return estimate_cbf_frequency(signal, fps=fps, min_hz=min_hz, max_hz=max_hz, method="fft")


def estimate_cbf_peaks(signal: np.ndarray, fps: float, min_hz: float = 3.0, max_hz: float = 40.0) -> dict:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    if len(signal) < 16:
        raise ValueError("Signal is too short for peak analysis.")

    try:
        min_hz, effective_max_hz, nyquist_hz = _effective_frequency_range(fps, min_hz, max_hz)
    except ValueError as exc:
        return {
            "method": "Peak interval",
            "cbf_hz": np.nan,
            "n_peaks": 0,
            "note": str(exc),
            "peaks": np.array([], dtype=int),
        }

    clean = detrend(np.asarray(signal, dtype=float))
    clean = clean - np.mean(clean)
    std = float(np.std(clean))
    if std <= 1e-12:
        return {
            "method": "Peak interval",
            "cbf_hz": np.nan,
            "n_peaks": 0,
            "note": "No measurable signal variation.",
            "peaks": np.array([], dtype=int),
        }

    min_distance_frames = max(1, int(fps / effective_max_hz))
    peaks, _props = find_peaks(clean, distance=min_distance_frames, prominence=std * 0.3)

    if len(peaks) < 2:
        return {
            "method": "Peak interval",
            "cbf_hz": np.nan,
            "n_peaks": int(len(peaks)),
            "note": "Too few peaks detected.",
            "peaks": peaks,
        }

    intervals_sec = np.diff(peaks) / fps
    freqs = 1.0 / intervals_sec
    valid = (freqs >= min_hz) & (freqs <= effective_max_hz)
    if not np.any(valid):
        return {
            "method": "Peak interval",
            "cbf_hz": np.nan,
            "n_peaks": int(len(peaks)),
            "note": "Detected peaks are outside selected frequency range.",
            "peaks": peaks,
        }

    return {
        "method": "Peak interval",
        "cbf_hz": float(np.median(freqs[valid])),
        "n_peaks": int(len(peaks)),
        "note": "Median peak-to-peak frequency.",
        "peaks": peaks,
    }


def make_kymograph(stack: np.ndarray, roi: ROI | None = None) -> np.ndarray:
    if roi is None:
        _, y_size, _x_size = stack.shape
        y_mid = y_size // 2
        return stack[:, y_mid, :]
    x, y, w, h = roi
    y_mid = y + h // 2
    return stack[:, y_mid, x : x + w]


def _block_mean_stack(stack: np.ndarray, tile_size: int) -> np.ndarray:
    """Downsample T,Y,X by block averaging in Y/X only."""
    if tile_size <= 1:
        return stack.astype(float, copy=False)
    t, y, x = stack.shape
    y_trim = (y // tile_size) * tile_size
    x_trim = (x // tile_size) * tile_size
    if y_trim < tile_size or x_trim < tile_size:
        raise ValueError("Tile size is too large for the selected region.")
    cropped = stack[:, :y_trim, :x_trim].astype(float, copy=False)
    return cropped.reshape(t, y_trim // tile_size, tile_size, x_trim // tile_size, tile_size).mean(axis=(2, 4))


def compute_cbf_heatmap(
    stack: np.ndarray,
    fps: float,
    min_hz: float = 3.0,
    max_hz: float = 40.0,
    method: str = "fft",
    tile_size: int = 8,
    roi: ROI | None = None,
) -> dict:
    """Compute dominant-frequency and peak-strength maps from video blocks.

    The returned maps are in tile coordinates. In napari, display them with
    ``scale=(tile_size, tile_size)`` and ``translate=(roi_y, roi_x)`` when ROI is used.
    """
    region = crop_stack_to_roi(stack, roi)
    block = _block_mean_stack(region, max(1, int(tile_size)))
    t, out_y, out_x = block.shape
    if t < 16:
        raise ValueError("Video is too short for heatmap frequency analysis.")

    cbf_map = np.full((out_y, out_x), np.nan, dtype=np.float32)
    strength_map = np.full((out_y, out_x), np.nan, dtype=np.float32)

    for yy in range(out_y):
        for xx in range(out_x):
            signal = block[:, yy, xx]
            try:
                result = estimate_cbf_frequency(signal, fps=fps, min_hz=min_hz, max_hz=max_hz, method=method)
            except Exception:
                continue
            cbf_map[yy, xx] = result["cbf_hz"]
            strength_map[yy, xx] = result["peak_to_background"]

    return {
        "cbf_map": cbf_map,
        "strength_map": strength_map,
        "tile_size": max(1, int(tile_size)),
        "roi": roi,
        "method": method,
    }


def compute_motion_activity_map(
    stack: np.ndarray,
    method: str = "temporal_sd",
    fps: float | None = None,
    min_hz: float = 3.0,
    max_hz: float = 40.0,
    tile_size: int = 4,
    roi: ROI | None = None,
) -> dict:
    """Compute simple maps that show where video motion/activity exists."""
    region = crop_stack_to_roi(stack, roi).astype(float, copy=False)
    method_key = method.strip().lower().replace(" ", "_").replace("-", "_")
    tile_size = max(1, int(tile_size))

    if method_key in {"temporal_sd", "temporal_std", "std"}:
        data = _block_mean_stack(region, tile_size)
        activity = np.std(data, axis=0)
        label = "Temporal SD"
    elif method_key in {"frame_difference", "frame_difference_mean", "diff"}:
        data = _block_mean_stack(region, tile_size)
        activity = np.mean(np.abs(np.diff(data, axis=0)), axis=0)
        label = "Mean frame difference"
    elif method_key in {"max_min", "range"}:
        data = _block_mean_stack(region, tile_size)
        activity = np.max(data, axis=0) - np.min(data, axis=0)
        label = "Max-min intensity range"
    elif method_key in {"band_limited_fft_power", "band_limited_power", "fft_power"}:
        if fps is None or fps <= 0:
            raise ValueError("FPS is required for band-limited FFT power.")
        data = _block_mean_stack(region, tile_size)
        clean = detrend(data, axis=0)
        clean = clean - np.mean(clean, axis=0, keepdims=True)
        freqs = np.fft.rfftfreq(clean.shape[0], d=1.0 / fps)
        power = np.abs(np.fft.rfft(clean * np.hanning(clean.shape[0])[:, None, None], axis=0)) ** 2
        _min_hz, effective_max_hz, _nyquist = _effective_frequency_range(fps, min_hz, max_hz)
        valid = (freqs >= min_hz) & (freqs <= effective_max_hz)
        if not np.any(valid):
            raise ValueError("No valid FFT bins in selected range.")
        activity = np.sum(power[valid, :, :], axis=0)
        label = "Band-limited FFT power"
    else:
        raise ValueError(f"Unsupported activity method: {method}")

    return {
        "activity_map": activity.astype(np.float32),
        "tile_size": tile_size,
        "roi": roi,
        "method": label,
    }


def _normalize_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    mn = float(np.min(frame))
    mx = float(np.max(frame))
    if mx <= mn:
        return np.zeros_like(frame, dtype=np.uint8)
    return np.clip((frame - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


def compute_optical_flow_maps(
    stack: np.ndarray,
    roi: ROI | None = None,
    frame_step: int = 1,
    max_pairs: int = 100,
) -> dict:
    """Compute average Farneback optical-flow descriptors.

    These maps describe apparent image motion. They should be treated as
    exploratory motion-field descriptors, not clinical beat-pattern classes.
    """
    region = crop_stack_to_roi(stack, roi)
    frame_step = max(1, int(frame_step))
    max_pairs = max(1, int(max_pairs))

    indices = list(range(0, region.shape[0] - frame_step, frame_step))[:max_pairs]
    if not indices:
        raise ValueError("Not enough frames for optical-flow analysis.")

    sum_u = None
    sum_v = None
    sum_mag = None
    n = 0

    for i in indices:
        prev = _normalize_uint8(region[i])
        nxt = _normalize_uint8(region[i + frame_step])
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u * u + v * v)
        if sum_u is None:
            sum_u = np.zeros_like(u, dtype=np.float64)
            sum_v = np.zeros_like(v, dtype=np.float64)
            sum_mag = np.zeros_like(mag, dtype=np.float64)
        sum_u += u
        sum_v += v
        sum_mag += mag
        n += 1

    mean_u = sum_u / n
    mean_v = sum_v / n
    mean_mag = sum_mag / n
    direction = np.arctan2(mean_v, mean_u)

    du_dy, du_dx = np.gradient(mean_u)
    dv_dy, dv_dx = np.gradient(mean_v)
    curl = dv_dx - du_dy
    deformation = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2)

    return {
        "magnitude": mean_mag.astype(np.float32),
        "direction": direction.astype(np.float32),
        "curl": curl.astype(np.float32),
        "deformation": deformation.astype(np.float32),
        "roi": roi,
        "frame_step": frame_step,
        "n_pairs": n,
    }
