from __future__ import annotations

"""Core numerical routines for napari-cilia-assistant.

Scientific scope
----------------
The functions in this module quantify rhythmic motion in high-speed cilia videos.
They are intentionally conservative and transparent: each step maps to a common
manual or computational ciliary beat frequency (CBF) workflow.

The analysis is based on three related measurements:

1. ROI mean-intensity signal
   A user-selected region of interest (ROI) is reduced to a 1-D time series by
   averaging pixel intensity within the ROI for every frame. Beating cilia cause
   periodic local intensity fluctuations as cilia move through the optical path.

2. FFT dominant-frequency estimate
   The detrended ROI signal is transformed into the frequency domain. The largest
   spectral peak within the user-defined physiological search range is reported
   as the automated CBF estimate.

3. Peak-interval estimate and kymograph
   The peak-interval method is an independent, semi-manual sanity check. The
   kymograph is a visual audit layer: each row is one frame and each column is
   position along a line across the moving cilia edge.

Important limitation
--------------------
This module estimates frequency, not full ciliary beat pattern or waveform. A
normal-looking CBF can still coexist with abnormal waveform. For publication or
collaborative reporting, the numerical CBF should therefore be interpreted with
ROI placement, video quality, frame rate, temperature, and visual/kymograph review.
"""

from contextlib import suppress

import cv2
import numpy as np
from scipy.signal import detrend, find_peaks


def read_avi_info(path: str) -> dict:
    """Read AVI metadata using OpenCV without loading the full movie.

    Why this exists
    ---------------
    CBF is measured in Hz, so the video frame rate is part of the scientific
    measurement. Reading and displaying FPS, frame count, duration, size, and
    codec gives the user a quick quality-control checkpoint before analysis.

    Notes
    -----
    OpenCV metadata can be wrong for some legacy AVI encoders. The UI therefore
    lets the user manually correct FPS after loading.
    """
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
        # Always release the handle. This matters on Windows, where an open
        # VideoCapture object can keep the AVI locked after a failed read.
        with suppress(Exception):
            cap.release()


def load_avi_as_stack(path: str, max_frames: int | None = None) -> np.ndarray:
    """Load an AVI video as a grayscale NumPy stack with shape ``T, Y, X``.

    Why grayscale
    -------------
    Most bright-field/DIC cilia videos encode motion as intensity changes, not as
    color information. Converting to grayscale gives a simple, reproducible input
    for ROI intensity, FFT, peak detection, and kymograph generation.

    Parameters
    ----------
    path:
        AVI file path.
    max_frames:
        Optional safety limit for long videos. ``None`` loads all frames.

    Returns
    -------
    stack:
        Three-dimensional array ordered as time, y, x.
    """
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

            # OpenCV returns color AVI frames as BGR. For CBF, color channels are
            # not needed; a single intensity image avoids channel-dependent bias.
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


def roi_from_shape_data(
    shape_data: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert napari Shapes vertices into a bounded rectangular ROI.

    Napari stores shape vertices in ``Y, X`` order. Many image-processing
    functions use ``x, y, width, height``. This conversion is deliberately kept in
    one place so downstream analysis receives a consistent ROI representation.

    Parameters
    ----------
    shape_data:
        Napari shape vertices, usually an ``N x 2`` array.
    image_shape:
        Image shape as ``Y, X``.

    Returns
    -------
    roi:
        ``x, y, width, height`` clipped to the image bounds.
    """
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

    # Clip to valid image bounds so ROI edits slightly outside the image do not
    # crash measurement. The minimum end coordinate preserves non-empty slices.
    y0 = max(0, min(y0, img_y - 1))
    y1 = max(1, min(y1, img_y))
    x0 = max(0, min(x0, img_x - 1))
    x1 = max(1, min(x1, img_x))

    width = x1 - x0
    height = y1 - y0

    if width <= 1 or height <= 1:
        raise ValueError("ROI is too small. Draw a larger rectangle over the moving cilia.")

    return x0, y0, width, height


def crop_stack_to_roi(
    stack: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return the full stack or an ``x, y, width, height`` crop.

    Keeping this crop as a view rather than a copy makes repeated ROI analysis
    lightweight for typical AVI stacks.
    """
    if roi is None:
        return stack

    x, y, w, h = roi
    return stack[:, y:y + h, x:x + w]


def roi_mean_signal(
    stack: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Compute mean ROI intensity for each video frame.

    Why mean intensity
    ------------------
    The mean intensity trace is a simple photoelectronic-style signal. It captures
    periodic brightness changes produced by cilia moving through the selected
    region, while averaging reduces single-pixel noise.

    Returns
    -------
    signal:
        One-dimensional array with length equal to the number of frames.
    """
    cropped = crop_stack_to_roi(stack, roi)
    return cropped.mean(axis=(1, 2)).astype(float)


def estimate_cbf_fft(
    signal: np.ndarray,
    fps: float,
    min_hz: float = 3.0,
    max_hz: float = 40.0,
) -> dict:
    """Estimate ciliary beat frequency from the dominant FFT peak.

    Method rationale
    ----------------
    A rhythmic cilia signal should contain repeated intensity oscillations. The
    FFT converts those oscillations into power versus frequency. The reported CBF
    is the strongest frequency peak within the user-defined search range.

    The preprocessing is intentionally modest:
    * detrending suppresses slow photobleaching, drift, or gradual focus change;
    * mean-centering removes the DC component;
    * a Hanning window reduces spectral leakage at the video boundaries.

    The function returns the full spectrum so reviewers/users can inspect whether
    the selected frequency is a sharp biological rhythm or a weak/noisy maximum.
    """
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    if len(signal) < 16:
        raise ValueError("Signal is too short for frequency analysis.")

    nyquist_hz = fps / 2.0
    effective_max_hz = min(max_hz, nyquist_hz * 0.95)

    if min_hz >= effective_max_hz:
        raise ValueError(
            f"Invalid frequency range. With FPS={fps}, max useful frequency is ~{nyquist_hz:.2f} Hz."
        )

    clean = detrend(signal)
    clean = clean - np.mean(clean)

    signal_std = float(np.std(clean))
    if signal_std <= 1e-12:
        raise ValueError("ROI signal has no intensity variation.")

    clean = clean * np.hanning(len(clean))

    freqs = np.fft.rfftfreq(len(clean), d=1.0 / fps)
    power = np.abs(np.fft.rfft(clean)) ** 2

    # Restrict to the expected biological/experimental range. This avoids
    # reporting low-frequency drift or high-frequency sensor noise as CBF.
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
        "method": "FFT dominant frequency",
        "cbf_hz": cbf_hz,
        "peak_to_background": peak_power / background_power,
        "freqs": freqs,
        "power": power,
        "effective_max_hz": effective_max_hz,
        "nyquist_hz": nyquist_hz,
    }


def estimate_cbf_peaks(
    signal: np.ndarray,
    fps: float,
    min_hz: float = 3.0,
    max_hz: float = 40.0,
) -> dict:
    """Estimate CBF from peak-to-peak spacing in the ROI intensity signal.

    This is not intended to replace the FFT estimate. It is an independent sanity
    check that resembles manual beat counting: repeated intensity maxima are
    treated as repeated beat cycles, and the median peak interval is converted to
    Hz. Disagreement between FFT and peak-interval values should trigger visual
    review of the ROI, kymograph, and raw movie.
    """
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    if len(signal) < 16:
        raise ValueError("Signal is too short for peak analysis.")

    nyquist_hz = fps / 2.0
    effective_max_hz = min(max_hz, nyquist_hz * 0.95)

    if min_hz >= effective_max_hz:
        return {
            "method": "Peak interval",
            "cbf_hz": np.nan,
            "n_peaks": 0,
            "note": f"Invalid frequency range for FPS={fps:.3f}; Nyquist is {nyquist_hz:.3f} Hz.",
            "peaks": np.array([], dtype=int),
        }

    clean = detrend(signal)
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

    # Enforce a minimum distance between peaks so a single beat is not counted
    # multiple times because of noise or substructure in the intensity trace.
    min_distance_frames = max(1, int(fps / effective_max_hz))
    prominence = std * 0.3

    peaks, _props = find_peaks(
        clean,
        distance=min_distance_frames,
        prominence=prominence,
    )

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

    cbf_hz = float(np.median(freqs[valid]))

    return {
        "method": "Peak interval",
        "cbf_hz": cbf_hz,
        "n_peaks": int(len(peaks)),
        "note": "Median peak-to-peak frequency.",
        "peaks": peaks,
    }


def make_kymograph(
    stack: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Create a horizontal-line kymograph from the stack.

    Output shape is ``T, X``. Each row is one time point, and each column is a
    spatial position along the middle row of the selected ROI. Repeating bands or
    waves in the kymograph provide a visual audit of periodic ciliary motion.

    Limitation
    ----------
    This is a deliberately simple center-line kymograph. For publication-grade
    waveform analysis, users should ensure that the ROI line crosses the moving
    cilia edge and should compare with raw video playback.
    """
    if roi is None:
        _, y_size, _x_size = stack.shape
        y_mid = y_size // 2
        return stack[:, y_mid, :]

    x, y, w, h = roi
    y_mid = y + h // 2

    return stack[:, y_mid, x:x + w]


def summarize_stack(stack: np.ndarray) -> dict:
    """Return basic stack statistics for UI logging and reproducibility."""
    return {
        "shape": tuple(stack.shape),
        "dtype": str(stack.dtype),
        "min": float(np.min(stack)),
        "max": float(np.max(stack)),
        "mean": float(np.mean(stack)),
    }
