from __future__ import annotations

"""napari Qt widget for ciliary beat frequency analysis.

The widget is intentionally workflow-oriented rather than fully automatic:
users open a high-speed AVI, confirm/adjust FPS, define an ROI over visibly
moving cilia, then compare FFT-derived CBF with a peak-interval sanity check
and a kymograph audit layer. This keeps the analysis transparent enough for
collaborative review and methods documentation.
"""

from pathlib import Path

import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ._analysis import (
    read_avi_info,
    load_avi_as_stack,
    summarize_stack,
    roi_from_shape_data,
    roi_mean_signal,
    estimate_cbf_fft,
    estimate_cbf_peaks,
    make_kymograph,
)


class CiliaAssistantWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.setObjectName("ciliaAssistant")

        self.viewer = napari_viewer
        self.video_path: str | None = None
        self.stack: np.ndarray | None = None
        self.info: dict | None = None

        self.roi_layer_name = "Cilia ROI"

        self.last_roi: tuple[int, int, int, int] | None = None
        self.last_signal: np.ndarray | None = None
        self.last_fft_result: dict | None = None
        self.last_peak_result: dict | None = None

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.title = QLabel("Cilia Assistant")
        self.title.setObjectName("appTitle")
        layout.addWidget(self.title)

        self.subtitle = QLabel("ROI-based beat-frequency measurement for high-speed AVI microscopy.")
        self.subtitle.setObjectName("appSubtitle")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        # -------------------------
        # Input section
        # -------------------------
        input_box = QGroupBox("1  Input")
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(12, 18, 12, 12)
        input_layout.setSpacing(8)

        self.open_button = QPushButton("Open AVI")
        self._style_button(self.open_button, "primary")
        self.open_button.setToolTip(
            "Load a high-speed AVI and inspect metadata before measuring CBF."
        )
        self.open_button.clicked.connect(self.open_avi)
        input_layout.addWidget(self.open_button)

        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(1, 5000)
        self.fps_box.setDecimals(3)
        self.fps_box.setValue(300.0)
        self.fps_box.setPrefix("FPS: ")
        self.fps_box.setToolTip(
            "Frame rate is required to convert frame intervals or FFT bins into Hz. "
            "Correct this manually if AVI metadata are wrong."
        )
        self.fps_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.fps_box)

        self.max_frames_box = QSpinBox()
        self.max_frames_box.setRange(0, 1000000)
        self.max_frames_box.setValue(1000)
        self.max_frames_box.setPrefix("Max frames, 0=all: ")
        self.max_frames_box.setToolTip(
            "Limits loading for very long videos. Use 0 to load the entire AVI."
        )
        self.max_frames_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.max_frames_box)

        input_box.setLayout(input_layout)
        layout.addWidget(input_box)

        # -------------------------
        # ROI section
        # -------------------------
        roi_box = QGroupBox("2  Region of Interest")
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(12, 18, 12, 12)
        roi_layout.setSpacing(8)

        self.create_roi_button = QPushButton("Create / Edit ROI Rectangle")
        self._style_button(self.create_roi_button, "secondary")
        self.create_roi_button.setToolTip(
            "Create a rectangle over an active ciliated edge; ROI placement is the main user-controlled scientific decision."
        )
        self.create_roi_button.clicked.connect(self.create_or_select_roi_layer)
        roi_layout.addWidget(self.create_roi_button)

        self.measure_roi_button = QPushButton("Measure CBF from Selected ROI")
        self._style_button(self.measure_roi_button, "primary")
        self.measure_roi_button.setToolTip(
            "Measure ROI mean-intensity oscillation, then estimate CBF using FFT and peak intervals."
        )
        self.measure_roi_button.clicked.connect(self.measure_selected_roi)
        roi_layout.addWidget(self.measure_roi_button)

        self.measure_whole_button = QPushButton("Measure Whole-Frame Motion Frequency")
        self._style_button(self.measure_whole_button, "secondary")
        self.measure_whole_button.setToolTip(
            "Exploratory only. Whole-frame signals may include illumination drift, stage motion, or tissue motion."
        )
        self.measure_whole_button.clicked.connect(self.measure_whole_frame)
        roi_layout.addWidget(self.measure_whole_button)

        roi_box.setLayout(roi_layout)
        layout.addWidget(roi_box)

        # -------------------------
        # Analysis parameters
        # -------------------------
        analysis_box = QGroupBox("3  Frequency Analysis")
        analysis_layout = QVBoxLayout()
        analysis_layout.setContentsMargins(12, 18, 12, 12)
        analysis_layout.setSpacing(8)

        self.min_hz_box = QDoubleSpinBox()
        self.min_hz_box.setRange(0.1, 500)
        self.min_hz_box.setDecimals(2)
        self.min_hz_box.setValue(3.0)
        self.min_hz_box.setPrefix("Min Hz: ")
        self.min_hz_box.setToolTip(
            "Lower bound for the accepted biological rhythm; helps reject slow drift."
        )
        self.min_hz_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analysis_layout.addWidget(self.min_hz_box)

        self.max_hz_box = QDoubleSpinBox()
        self.max_hz_box.setRange(0.1, 500)
        self.max_hz_box.setDecimals(2)
        self.max_hz_box.setValue(25.0)
        self.max_hz_box.setPrefix("Max Hz: ")
        self.max_hz_box.setToolTip(
            "Upper bound for CBF search. The code also respects the Nyquist limit from FPS."
        )
        self.max_hz_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analysis_layout.addWidget(self.max_hz_box)

        self.kymo_button = QPushButton("Create Kymograph from Selected ROI")
        self._style_button(self.kymo_button, "secondary")
        self.kymo_button.setToolTip(
            "Create a time-vs-position audit image to visually confirm rhythmic cilia motion."
        )
        self.kymo_button.clicked.connect(self.create_kymograph_layer)
        analysis_layout.addWidget(self.kymo_button)

        self.export_button = QPushButton("Export Last ROI Signal + FFT CSV")
        self._style_button(self.export_button, "secondary")
        self.export_button.setToolTip(
            "Export raw time-intensity and FFT power data for methods documentation or re-analysis."
        )
        self.export_button.clicked.connect(self.export_last_measurement)
        analysis_layout.addWidget(self.export_button)

        analysis_box.setLayout(analysis_layout)
        layout.addWidget(analysis_box)

        # -------------------------
        # Plot section
        # -------------------------
        plot_box = QGroupBox("4  Measurement Graph")
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(12, 18, 12, 12)
        plot_layout.setSpacing(8)

        self.figure = Figure(figsize=(5, 4))
        self.figure.patch.set_facecolor("#111827")
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        plot_button_row = QHBoxLayout()

        self.copy_graph_button = QPushButton("Copy Graphic")
        self._style_button(self.copy_graph_button, "secondary")
        self.copy_graph_button.setToolTip(
            "Copy the current measurement graph to the system clipboard as an image."
        )
        self.copy_graph_button.clicked.connect(self.copy_measurement_graph)
        plot_button_row.addWidget(self.copy_graph_button)

        self.save_graph_button = QPushButton("Save Graphic as TIFF")
        self._style_button(self.save_graph_button, "secondary")
        self.save_graph_button.setToolTip(
            "Save the current measurement graph as a TIFF image."
        )
        self.save_graph_button.clicked.connect(self.save_measurement_graph_as_tiff)
        plot_button_row.addWidget(self.save_graph_button)

        plot_layout.addLayout(plot_button_row)
        plot_box.setLayout(plot_layout)
        layout.addWidget(plot_box)

        # -------------------------
        # Log section
        # -------------------------
        log_box = QGroupBox("5  Log")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(12, 18, 12, 12)
        log_layout.setSpacing(8)

        self.clear_log_button = QPushButton("Clear Log")
        self._style_button(self.clear_log_button, "secondary")
        self.clear_log_button.setToolTip(
            "Clear previous messages before opening or measuring the next video."
        )
        self.clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_button)

        self.output = QTextEdit()
        self.output.setObjectName("runLog")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Measurement messages, metadata, and QC notes will appear here.")
        log_layout.addWidget(self.output)

        log_box.setLayout(log_layout)
        layout.addWidget(log_box)

        self.setLayout(layout)
        self._apply_ux_theme()

    def _style_button(self, button: QPushButton, variant: str):
        button.setProperty("variant", variant)
        button.setCursor(Qt.PointingHandCursor)

    def _apply_ux_theme(self):
        self.setStyleSheet(
            """
            QWidget#ciliaAssistant {
                background: #0f172a;
                color: #e5e7eb;
                font-size: 13px;
            }

            QLabel#appTitle {
                color: #f8fafc;
                font-size: 20px;
                font-weight: 700;
                padding: 2px 0 0 0;
            }

            QLabel#appSubtitle {
                color: #a7b1c2;
                padding: 0 0 4px 0;
            }

            QGroupBox {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 700;
                color: #f9fafb;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 3px 9px;
                border-radius: 7px;
                background: #334155;
                color: #f8fafc;
            }

            QPushButton {
                min-height: 30px;
                border-radius: 7px;
                padding: 6px 10px;
                font-weight: 650;
            }

            QPushButton[variant="primary"] {
                background: #2563eb;
                border: 1px solid #3b82f6;
                color: #ffffff;
            }

            QPushButton[variant="primary"]:hover {
                background: #1d4ed8;
                border-color: #60a5fa;
            }

            QPushButton[variant="secondary"] {
                background: #273244;
                border: 1px solid #475569;
                color: #e5e7eb;
            }

            QPushButton[variant="secondary"]:hover {
                background: #334155;
                border-color: #64748b;
            }

            QPushButton:pressed {
                padding-top: 7px;
                padding-bottom: 5px;
            }

            QPushButton:disabled {
                background: #1f2937;
                border-color: #334155;
                color: #6b7280;
            }

            QSpinBox,
            QDoubleSpinBox,
            QTextEdit {
                background: #0b1220;
                border: 1px solid #334155;
                border-radius: 7px;
                color: #e5e7eb;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }

            QSpinBox,
            QDoubleSpinBox {
                min-height: 28px;
                padding: 3px 8px;
            }

            QSpinBox:focus,
            QDoubleSpinBox:focus,
            QTextEdit:focus {
                border-color: #60a5fa;
            }

            QTextEdit#runLog {
                min-height: 120px;
                padding: 8px;
                font-family: monospace;
            }
            """
        )

    def _style_plot_axes(self, axes):
        axes.set_facecolor("#0b1220")
        axes.tick_params(colors="#cbd5e1")
        axes.xaxis.label.set_color("#dbeafe")
        axes.yaxis.label.set_color("#dbeafe")
        axes.title.set_color("#f8fafc")
        for spine in axes.spines.values():
            spine.set_color("#334155")
        axes.grid(True, color="#334155", alpha=0.35, linewidth=0.7)

    # -------------------------
    # Logging
    # -------------------------
    def log(self, text: str):
        self.output.append(text)

    def clear_log(self):
        self.output.clear()
        self.log("Log cleared.")

    # -------------------------
    # Video loading
    # -------------------------
    def open_avi(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open AVI video",
            "",
            "AVI files (*.avi);;All files (*)",
        )

        if not path:
            return

        self.video_path = path

        try:
            self.info = read_avi_info(path)

            fps = self.info.get("fps", 0)
            if fps and fps > 0:
                self.fps_box.setValue(float(fps))

            max_frames = self.max_frames_box.value()
            max_frames = None if max_frames == 0 else max_frames

            self.stack = load_avi_as_stack(path, max_frames=max_frames)

        except Exception as e:
            self.log("\nFailed to open AVI:")
            self.log(str(e))
            return

        layer_name = Path(path).stem

        self.viewer.add_image(
            self.stack,
            name=f"{layer_name}_avi_stack",
            colormap="gray",
            contrast_limits=[float(self.stack.min()), float(self.stack.max())],
        )

        self.log("\nOpened AVI:")
        for key, value in self.info.items():
            self.log(f"  {key}: {value}")

        stats = summarize_stack(self.stack)
        self.log("\nLoaded stack:")
        for key, value in stats.items():
            self.log(f"  {key}: {value}")

        self.log("\nNext step: click 'Create / Edit ROI Rectangle', place it over moving cilia, then measure ROI.")

    # -------------------------
    # ROI handling
    # -------------------------
    def _find_roi_layer(self):
        for layer in self.viewer.layers:
            if layer.name == self.roi_layer_name:
                return layer
        return None

    def create_or_select_roi_layer(self):
        if self.stack is None:
            self.log("Load an AVI first.")
            return

        existing = self._find_roi_layer()

        if existing is not None:
            self.viewer.layers.selection.active = existing
            existing.mode = "select"
            self.log("Selected existing ROI layer. Move or resize the rectangle over active cilia.")
            return

        _, y_size, x_size = self.stack.shape

        # Default rectangle in the center. The user should move it to the
        # active ciliated edge; ROI placement determines whether the measured
        # signal reflects cilia motion or unrelated tissue/illumination changes.
        y0 = int(y_size * 0.40)
        y1 = int(y_size * 0.60)
        x0 = int(x_size * 0.40)
        x1 = int(x_size * 0.60)

        rectangle = np.array(
            [
                [y0, x0],
                [y0, x1],
                [y1, x1],
                [y1, x0],
            ]
        )

        roi_layer = self.viewer.add_shapes(
            [rectangle],
            shape_type=["rectangle"],
            name=self.roi_layer_name,
            edge_color="yellow",
            face_color=[1, 1, 0, 0.10],
            edge_width=2,
        )

        roi_layer.mode = "select"
        self.viewer.layers.selection.active = roi_layer

        self.log("Created ROI rectangle. Move/resize it over the moving cilia edge, then click 'Measure CBF from Selected ROI'.")

    def _get_selected_roi(self) -> tuple[int, int, int, int]:
        if self.stack is None:
            raise ValueError("No AVI stack loaded.")

        roi_layer = self._find_roi_layer()

        if roi_layer is None:
            raise ValueError("No ROI layer found. Click 'Create / Edit ROI Rectangle' first.")

        if len(roi_layer.data) == 0:
            raise ValueError("ROI layer exists but contains no shape.")

        selected = list(roi_layer.selected_data)

        if selected:
            shape_index = selected[0]
        else:
            # Use the most recent shape if nothing is selected. This makes the
            # workflow forgiving when the user draws a rectangle and immediately
            # clicks Measure without explicitly selecting it.
            shape_index = len(roi_layer.data) - 1

        shape_data = roi_layer.data[shape_index]

        roi = roi_from_shape_data(
            shape_data=shape_data,
            image_shape=self.stack.shape[1:],
        )

        return roi

    # -------------------------
    # Measurement
    # -------------------------
    def measure_selected_roi(self):
        try:
            roi = self._get_selected_roi()
            self._run_measurement(roi=roi, label="Selected ROI")
        except Exception as e:
            self.log("\nROI measurement failed:")
            self.log(str(e))

    def measure_whole_frame(self):
        try:
            self._run_measurement(roi=None, label="Whole frame")
        except Exception as e:
            self.log("\nWhole-frame measurement failed:")
            self.log(str(e))

    def _run_measurement(
        self,
        roi: tuple[int, int, int, int] | None,
        label: str,
    ):
        if self.stack is None:
            raise ValueError("No AVI stack loaded.")

        fps = float(self.fps_box.value())
        min_hz = float(self.min_hz_box.value())
        max_hz = float(self.max_hz_box.value())

        # Convert the selected video region into a single temporal signal.
        # This intentionally mirrors classical light-intensity fluctuation
        # approaches, while keeping the ROI visible and editable in napari.
        signal = roi_mean_signal(self.stack, roi=roi)

        # FFT is the primary automated estimate; peak intervals provide an
        # independent check against obvious failure modes such as weak/noisy ROIs.
        fft_result = estimate_cbf_fft(
            signal=signal,
            fps=fps,
            min_hz=min_hz,
            max_hz=max_hz,
        )

        peak_result = estimate_cbf_peaks(
            signal=signal,
            fps=fps,
            min_hz=min_hz,
            max_hz=max_hz,
        )

        self.last_roi = roi
        self.last_signal = signal
        self.last_fft_result = fft_result
        self.last_peak_result = peak_result

        self.log("\nCBF measurement:")
        self.log(f"  Region: {label}")

        if roi is None:
            self.log("  ROI: whole frame")
            self.log("  Warning: whole-frame analysis may detect illumination/tissue motion, not true cilia.")
            self.log("  For publication-style CBF, prefer a small ROI over visibly beating cilia.")
        else:
            x, y, w, h = roi
            self.log(f"  ROI: x={x}, y={y}, width={w}, height={h}")

        self.log(f"  FPS used: {fps:.3f}")
        self.log(f"  Search range: {min_hz:.2f}–{fft_result['effective_max_hz']:.2f} Hz")
        self.log(f"  Nyquist limit: {fft_result['nyquist_hz']:.2f} Hz")

        self.log(f"  FFT CBF: {fft_result['cbf_hz']:.3f} Hz")
        self.log(f"  FFT peak/background: {fft_result['peak_to_background']:.2f}")

        peak_cbf = peak_result.get("cbf_hz", np.nan)
        if np.isfinite(peak_cbf):
            self.log(f"  Peak-interval CBF: {peak_cbf:.3f} Hz")
            self.log(f"  Detected peaks: {peak_result.get('n_peaks', 0)}")
        else:
            self.log("  Peak-interval CBF: not reliable")
            self.log(f"  Peak note: {peak_result.get('note', 'N/A')}")

        self._plot_signal_and_fft(signal, fft_result, peak_result, fps, label)

    def _plot_signal_and_fft(
        self,
        signal: np.ndarray,
        fft_result: dict,
        peak_result: dict,
        fps: float,
        label: str,
    ):
        self.figure.clear()

        time = np.arange(len(signal)) / fps

        ax1 = self.figure.add_subplot(2, 1, 1)
        self._style_plot_axes(ax1)
        ax1.plot(time, signal, color="#60a5fa", linewidth=1.6)
        ax1.set_title(f"{label}: ROI mean intensity vs time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mean intensity")

        peaks = peak_result.get("peaks", np.array([], dtype=int))
        if peaks is not None and len(peaks) > 0:
            valid_peaks = peaks[(peaks >= 0) & (peaks < len(signal))]
            ax1.plot(
                time[valid_peaks],
                signal[valid_peaks],
                "o",
                color="#f59e0b",
                markersize=3,
            )

        ax2 = self.figure.add_subplot(2, 1, 2)
        self._style_plot_axes(ax2)
        freqs = fft_result["freqs"]
        power = fft_result["power"]
        cbf_hz = fft_result["cbf_hz"]

        ax2.plot(freqs, power, color="#22c55e", linewidth=1.6)
        # Mark the selected spectral peak so the reviewer/user can judge whether
        # the reported CBF corresponds to a clear dominant rhythm.
        ax2.axvline(cbf_hz, linestyle="--", color="#f59e0b", linewidth=1.3)
        ax2.set_xlim(0, min(fft_result["effective_max_hz"] * 1.2, fps / 2.0))
        ax2.set_title(f"FFT spectrum: dominant rhythm = {cbf_hz:.3f} Hz")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("FFT power")

        self.figure.tight_layout()
        self.canvas.draw()

    # -------------------------
    # Measurement graphic helpers
    # -------------------------
    def _has_measurement_graph(self) -> bool:
        return self.last_signal is not None and len(self.figure.axes) > 0

    def copy_measurement_graph(self):
        if not self._has_measurement_graph():
            self.log("No measurement graph available to copy yet.")
            return

        try:
            pixmap = self.canvas.grab()
            if pixmap.isNull():
                raise RuntimeError("Could not capture the graph canvas.")
            QApplication.clipboard().setPixmap(pixmap)
            self.log("Copied measurement graphic to clipboard.")
        except Exception as e:
            self.log("Failed to copy measurement graphic:")
            self.log(str(e))

    def save_measurement_graph_as_tiff(self):
        if not self._has_measurement_graph():
            self.log("No measurement graph available to save yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Measurement Graphic as TIFF",
            "cilia_measurement_graph.tiff",
            "TIFF files (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return

        try:
            self.figure.savefig(path, format="tiff", dpi=300, bbox_inches="tight")
            self.log(f"Saved measurement graphic: {path}")
        except Exception as e:
            self.log("Failed to save measurement graphic:")
            self.log(str(e))

    # -------------------------
    # Kymograph
    # -------------------------
    def create_kymograph_layer(self):
        if self.stack is None:
            self.log("No AVI stack loaded.")
            return

        try:
            roi = self._get_selected_roi()
            kymo = make_kymograph(self.stack, roi=roi)
        except Exception as e:
            self.log("\nCould not create selected-ROI kymograph:")
            self.log(str(e))
            self.log("Creating whole-frame center-line kymograph instead.")
            kymo = make_kymograph(self.stack, roi=None)

        self.viewer.add_image(
            kymo,
            name="cilia_kymograph_time_vs_position",
            colormap="gray",
            contrast_limits=[float(kymo.min()), float(kymo.max())],
        )

        self.log("\nCreated kymograph layer.")
        self.log("  Kymograph shape: T x X-position")
        self.log("  Each row is one video frame.")
        self.log("  Repeated bands/waves indicate rhythmic cilia motion.")
        self.log("  Use this as a visual audit; CBF alone does not prove normal waveform.")

    # -------------------------
    # Export
    # -------------------------
    def export_last_measurement(self):
        if self.last_signal is None or self.last_fft_result is None:
            self.log("No measurement to export yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cilia Measurement CSV",
            "cilia_roi_measurement.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        fps = float(self.fps_box.value())
        time = np.arange(len(self.last_signal)) / fps

        signal_path = Path(path)
        fft_path = signal_path.with_name(signal_path.stem + "_fft.csv")

        signal_table = np.column_stack([time, self.last_signal])
        np.savetxt(
            signal_path,
            signal_table,
            delimiter=",",
            header="time_sec,roi_mean_intensity",
            comments="",
        )

        freqs = self.last_fft_result["freqs"]
        power = self.last_fft_result["power"]
        fft_table = np.column_stack([freqs, power])
        np.savetxt(
            fft_path,
            fft_table,
            delimiter=",",
            header="frequency_hz,fft_power",
            comments="",
        )

        self.log("\nExported measurement:")
        self.log(f"  Signal CSV: {signal_path}")
        self.log(f"  FFT CSV: {fft_path}")
