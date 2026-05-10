from __future__ import annotations

"""Main napari Qt widget for ciliary beat-frequency and motion analysis.

The UI is intentionally split into five workflow steps. Step 3 contains analysis
sub-tabs so users can choose the right level of analysis without overcrowding
the standard ROI CBF workflow.
"""

from pathlib import Path

import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QCheckBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QTabWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ._analysis import (
    read_avi_info,
    load_avi_as_stack,
    summarize_stack,
    roi_from_shape_data,
    roi_mean_signal,
    estimate_cbf_frequency,
    estimate_cbf_peaks,
    make_kymograph,
    compute_cbf_heatmap,
    compute_motion_activity_map,
    compute_optical_flow_maps,
)
from .tabs import RoiFrequencyTab, CbfHeatmapTab, MotionActivityTab, AdvancedFlowTab


class CollapsiblePanel(QWidget):
    def __init__(self, title: str, content: QWidget, collapsed: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.content = content
        step_text, title_text = self._split_title(title)

        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(collapsed)
        self.toggle_button.setText("+" if collapsed else "−")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setObjectName("collapsibleToggle")
        self.toggle_button.clicked.connect(self._on_toggled)

        self.step_label = QLabel(step_text)
        self.step_label.setObjectName("collapsibleStepBadge")
        self.step_label.setVisible(bool(step_text))

        self.title_label = QLabel(title_text)
        self.title_label.setObjectName("collapsibleTitle")

        self.header_line = QFrame()
        self.header_line.setFrameShape(QFrame.HLine)
        self.header_line.setFrameShadow(QFrame.Sunken)
        self.header_line.setObjectName("collapsibleHeaderLine")

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.step_label)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.header_line, 1)

        self.body_frame = QFrame()
        self.body_frame.setObjectName("collapsibleBody")
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)
        body_layout.addWidget(self.content)
        self.body_frame.setLayout(body_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header_layout)
        layout.addWidget(self.body_frame)
        self.setLayout(layout)
        self._on_toggled(collapsed)

    def _split_title(self, title: str) -> tuple[str, str]:
        head, separator, tail = title.strip().partition(".")
        if separator and head.strip() and tail.strip():
            return head.strip(), tail.strip()
        return "", title.strip()

    def _on_toggled(self, checked: bool) -> None:
        self.toggle_button.setText("+" if checked else "−")
        self.body_frame.setVisible(not checked)
        self.updateGeometry()


class CiliaAssistantWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.setObjectName("ciliaAssistant")
        self.setMinimumWidth(380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.viewer = napari_viewer
        self.video_path: str | None = None
        self.stack: np.ndarray | None = None
        self.info: dict | None = None

        self.roi_layer_name = "Cilia ROI"
        self.background_roi_layer_name = "Cilia Background ROI"

        self.last_roi: tuple[int, int, int, int] | None = None
        self.last_background_roi: tuple[int, int, int, int] | None = None
        self.last_signal: np.ndarray | None = None
        self.last_raw_signal: np.ndarray | None = None
        self.last_background_signal: np.ndarray | None = None
        self.last_frequency_result: dict | None = None
        self.last_peak_result: dict | None = None
        self.last_map_result: dict | None = None

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_header())

        scroll_area = QScrollArea()
        scroll_area.setObjectName("assistantScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll_content = QWidget()
        scroll_content.setObjectName("assistantScrollContent")
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        content_layout.addWidget(CollapsiblePanel("Results. Key Measurements", self._build_results_box(), collapsed=False))
        content_layout.addWidget(CollapsiblePanel("Step 1. Input", self._build_input_box(), collapsed=False))
        content_layout.addWidget(CollapsiblePanel("Step 2. Region of Interest", self._build_roi_box(), collapsed=False))
        content_layout.addWidget(CollapsiblePanel("Step 3. Analysis", self._build_analysis_tabs(), collapsed=False))
        content_layout.addWidget(CollapsiblePanel("Step 4. Results / Graphs", self._build_plot_box(), collapsed=False), 1)
        content_layout.addWidget(CollapsiblePanel("Step 5. Export & Log", self._build_export_log_box(), collapsed=True), 1)

        scroll_content.setLayout(content_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        self.setLayout(layout)
        self._apply_ux_theme()

    # -------------------------
    # UI builders
    # -------------------------
    def _build_header(self) -> QWidget:
        header_frame = QFrame()
        header_frame.setObjectName("assistantHeader")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(12)

        logo = QLabel("CBF")
        logo.setObjectName("appLogo")
        logo.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(2)
        title = QLabel("Cilia Assistant")
        title.setObjectName("appTitle")
        title_column.addWidget(title)
        subtitle = QLabel("ROI CBF, spatial heatmaps, activity maps, and exploratory flow analysis")
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column, 1)
        header_frame.setLayout(header_layout)
        return header_frame

    def _build_results_box(self) -> QWidget:
        results_box = QWidget()
        results_layout = QHBoxLayout()
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        self.frequency_result_card = self._build_result_card("Frequency CBF", "—", "Primary estimate")
        self.peak_result_card = self._build_result_card("Peak CBF", "—", "Independent check")
        results_layout.addWidget(self.frequency_result_card)
        results_layout.addWidget(self.peak_result_card)
        results_box.setLayout(results_layout)
        return results_box

    def _build_input_box(self) -> QWidget:
        input_box = QWidget()
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self.open_button = QPushButton("Open AVI")
        self._style_button(self.open_button, "primary")
        self.open_button.setToolTip("Load a high-speed AVI and inspect metadata before analysis.")
        self.open_button.clicked.connect(self.open_avi)
        input_layout.addWidget(self.open_button)

        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(1, 5000)
        self.fps_box.setDecimals(3)
        self.fps_box.setValue(300.0)
        self.fps_box.setPrefix("FPS: ")
        self.fps_box.setToolTip("Correct this manually if AVI metadata are wrong.")
        input_layout.addWidget(self.fps_box)

        self.max_frames_box = QSpinBox()
        self.max_frames_box.setRange(0, 1000000)
        self.max_frames_box.setValue(1000)
        self.max_frames_box.setPrefix("Max frames, 0=all: ")
        self.max_frames_box.setToolTip("Use 0 to load the entire AVI. Start with a limit for very long files.")
        input_layout.addWidget(self.max_frames_box)

        input_box.setLayout(input_layout)
        return input_box

    def _build_roi_box(self) -> QWidget:
        roi_box = QWidget()
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(8)

        self.create_roi_button = QPushButton("Create / Edit ROI Rectangle")
        self._style_button(self.create_roi_button, "secondary")
        self.create_roi_button.setToolTip("Create a rectangle over active cilia. ROI placement is the main scientific decision.")
        self.create_roi_button.clicked.connect(self.create_or_select_roi_layer)
        roi_layout.addWidget(self.create_roi_button)

        self.create_background_button = QPushButton("Create / Edit Background ROI")
        self._style_button(self.create_background_button, "secondary")
        self.create_background_button.setToolTip("Optional nearby non-cilia region for subtracting shared illumination or focus drift.")
        self.create_background_button.clicked.connect(self.create_or_select_background_roi_layer)
        roi_layout.addWidget(self.create_background_button)

        self.subtract_background_check = QCheckBox("Subtract background ROI for ROI frequency analysis")
        self.subtract_background_check.setChecked(True)
        roi_layout.addWidget(self.subtract_background_check)

        roi_box.setLayout(roi_layout)
        return roi_box

    def _build_analysis_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setObjectName("analysisTabs")
        tabs.addTab(RoiFrequencyTab(self), "ROI Frequency")
        tabs.addTab(CbfHeatmapTab(self), "CBF Heatmap")
        tabs.addTab(MotionActivityTab(self), "Motion Activity")
        tabs.addTab(AdvancedFlowTab(self), "Advanced Flow")
        return tabs

    def _build_plot_box(self) -> QWidget:
        plot_box = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(8)

        self.figure = Figure(figsize=(5, 3.4))
        self.figure.patch.set_facecolor("#070d18")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(380)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas)

        row = QHBoxLayout()
        self.copy_graph_button = QPushButton("Copy Graphic")
        self._style_button(self.copy_graph_button, "secondary")
        self.copy_graph_button.clicked.connect(self.copy_measurement_graph)
        row.addWidget(self.copy_graph_button)

        self.save_graph_button = QPushButton("Save Graphic as TIFF")
        self._style_button(self.save_graph_button, "secondary")
        self.save_graph_button.clicked.connect(self.save_measurement_graph_as_tiff)
        row.addWidget(self.save_graph_button)
        plot_layout.addLayout(row)

        plot_box.setLayout(plot_layout)
        return plot_box

    def _build_export_log_box(self) -> QWidget:
        log_box = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(8)

        row = QHBoxLayout()
        self.export_button = QPushButton("Export Last Analysis")
        self._style_button(self.export_button, "secondary")
        self.export_button.clicked.connect(self.export_last_analysis)
        row.addWidget(self.export_button)

        self.copy_log_button = QPushButton("Copy Log")
        self._style_button(self.copy_log_button, "secondary")
        self.copy_log_button.clicked.connect(self.copy_log)
        row.addWidget(self.copy_log_button)

        self.clear_log_button = QPushButton("Clear Log")
        self._style_button(self.clear_log_button, "secondary")
        self.clear_log_button.clicked.connect(self.clear_log)
        row.addWidget(self.clear_log_button)
        log_layout.addLayout(row)

        self.output = QTextEdit()
        self.output.setObjectName("runLog")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Measurement messages, metadata, and QC notes will appear here.")
        self.output.setMinimumHeight(150)
        log_layout.addWidget(self.output)

        log_box.setLayout(log_layout)
        return log_box

    def _build_result_card(self, label: str, value: str, note: str) -> QFrame:
        card = QFrame()
        card.setObjectName("resultCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        title = QLabel(label)
        title.setObjectName("resultLabel")
        layout.addWidget(title)
        value_label = QLabel(value)
        value_label.setObjectName("resultValue")
        layout.addWidget(value_label)
        note_label = QLabel(note)
        note_label.setObjectName("resultNote")
        layout.addWidget(note_label)
        card.setLayout(layout)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return card

    def _set_result_card_value(self, card: QFrame, value: str, note: str | None = None):
        labels = card.findChildren(QLabel)
        if len(labels) >= 2:
            labels[1].setText(value)
        if note is not None and len(labels) >= 3:
            labels[2].setText(note)

    def _style_button(self, button: QPushButton, variant: str):
        button.setProperty("variant", variant)
        button.setCursor(Qt.PointingHandCursor)

    # -------------------------
    # Logging
    # -------------------------
    def log(self, text: str):
        self.output.append(text)

    def clear_log(self):
        self.output.clear()
        self.log("Log cleared.")

    def _log_export_text(self) -> str:
        lines = ["Cilia Assistant Log"]
        if self.video_path:
            path = Path(self.video_path)
            lines.extend([f"Current file name: {path.name}", f"Current file path: {path}"])
        else:
            lines.append("Current file name: not loaded")
        if self.info:
            lines.extend([
                f"Metadata FPS: {self.info.get('fps', 'unknown')}",
                f"Metadata frames: {self.info.get('frame_count', 'unknown')}",
                f"Metadata duration_sec: {self.info.get('duration_sec', 'unknown')}",
            ])
        log_text = self.output.toPlainText().strip()
        lines.extend(["", "Log:", log_text if log_text else "(empty)"])
        return "\n".join(lines)

    def copy_log(self):
        QApplication.clipboard().setText(self._log_export_text())
        self.log("Copied log to clipboard with video file information.")

    # -------------------------
    # Video and ROI handling
    # -------------------------
    def open_avi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open AVI video", "", "AVI files (*.avi);;All files (*)")
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
        except Exception as exc:
            self.log("\nFailed to open AVI:")
            self.log(str(exc))
            return

        layer_name = Path(path).stem
        self.viewer.add_image(
            self.stack,
            name=f"{layer_name}_avi_stack",
            colormap="gray",
            contrast_limits=[float(self.stack.min()), float(self.stack.max())],
        )

        self.log("\nOpened AVI:")
        self.log(f"  File name: {Path(path).name}")
        self.log(f"  File path: {path}")
        for key, value in self.info.items():
            self.log(f"  {key}: {value}")
        self.log("\nLoaded stack:")
        for key, value in summarize_stack(self.stack).items():
            self.log(f"  {key}: {value}")
        self.log("\nNext: create/edit ROI, then choose an analysis tab in Step 3.")

    def _find_layer(self, layer_name: str):
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def create_or_select_roi_layer(self):
        self._create_or_select_rectangle_layer(
            self.roi_layer_name, "ROI", "yellow", [1, 1, 0, 0.10], (0.40, 0.60), (0.40, 0.60)
        )

    def create_or_select_background_roi_layer(self):
        self._create_or_select_rectangle_layer(
            self.background_roi_layer_name, "background ROI", "cyan", [0, 1, 1, 0.08], (0.08, 0.22), (0.40, 0.60)
        )

    def _create_or_select_rectangle_layer(
        self,
        layer_name: str,
        log_name: str,
        edge_color: str,
        face_color,
        y_bounds: tuple[float, float],
        x_bounds: tuple[float, float],
    ):
        if self.stack is None:
            self.log("Load an AVI first.")
            return
        existing = self._find_layer(layer_name)
        if existing is not None:
            self.viewer.layers.selection.active = existing
            existing.mode = "select"
            self.log(f"Selected existing {log_name} layer. Move or resize the rectangle as needed.")
            return

        _, y_size, x_size = self.stack.shape
        y0, y1 = int(y_size * y_bounds[0]), int(y_size * y_bounds[1])
        x0, x1 = int(x_size * x_bounds[0]), int(x_size * x_bounds[1])
        rectangle = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
        roi_layer = self.viewer.add_shapes(
            [rectangle],
            shape_type=["rectangle"],
            name=layer_name,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=2,
        )
        roi_layer.mode = "select"
        self.viewer.layers.selection.active = roi_layer
        self.log(f"Created {log_name} rectangle. Move/resize it before analysis.")

    def _get_roi_from_layer(self, layer_name: str, missing_message: str) -> tuple[int, int, int, int]:
        if self.stack is None:
            raise ValueError("No AVI stack loaded.")
        roi_layer = self._find_layer(layer_name)
        if roi_layer is None:
            raise ValueError(missing_message)
        if len(roi_layer.data) == 0:
            raise ValueError(f"{layer_name} layer exists but contains no shape.")
        selected = list(roi_layer.selected_data)
        shape_index = selected[0] if selected else len(roi_layer.data) - 1
        return roi_from_shape_data(roi_layer.data[shape_index], image_shape=self.stack.shape[1:])

    def _get_selected_roi(self) -> tuple[int, int, int, int]:
        return self._get_roi_from_layer(self.roi_layer_name, "No ROI layer found. Click 'Create / Edit ROI Rectangle' first.")

    def _get_background_roi(self) -> tuple[int, int, int, int]:
        return self._get_roi_from_layer(self.background_roi_layer_name, "No background ROI layer found.")

    def _roi_for_region_mode(self, region_mode: str) -> tuple[int, int, int, int] | None:
        return None if region_mode.lower().startswith("whole") else self._get_selected_roi()

    def _map_layer_placement(self, roi: tuple[int, int, int, int] | None, tile_size: int = 1):
        if roi is None:
            return (0, 0), (tile_size, tile_size)
        x, y, _w, _h = roi
        return (y, x), (tile_size, tile_size)

    # -------------------------
    # Analysis actions called by tabs
    # -------------------------
    def run_roi_frequency(self, method: str, min_hz: float, max_hz: float, whole_frame: bool = False):
        try:
            if self.stack is None:
                raise ValueError("No AVI stack loaded.")
            roi = None if whole_frame else self._get_selected_roi()
            fps = float(self.fps_box.value())
            raw_signal = roi_mean_signal(self.stack, roi=roi)
            signal = raw_signal
            background_roi = None
            background_signal = None

            if roi is not None and self.subtract_background_check.isChecked():
                background_layer = self._find_layer(self.background_roi_layer_name)
                if background_layer is not None and len(background_layer.data) > 0:
                    background_roi = self._get_background_roi()
                    background_signal = roi_mean_signal(self.stack, roi=background_roi)
                    signal = raw_signal - background_signal

            frequency_result = estimate_cbf_frequency(signal, fps=fps, min_hz=min_hz, max_hz=max_hz, method=method)
            peak_result = estimate_cbf_peaks(signal, fps=fps, min_hz=min_hz, max_hz=max_hz)

            self.last_roi = roi
            self.last_background_roi = background_roi
            self.last_raw_signal = raw_signal
            self.last_background_signal = background_signal
            self.last_signal = signal
            self.last_frequency_result = frequency_result
            self.last_peak_result = peak_result
            self.last_map_result = None

            freq_value = frequency_result["cbf_hz"]
            self._set_result_card_value(self.frequency_result_card, f"{freq_value:.3f} Hz", method)
            if np.isfinite(peak_result.get("cbf_hz", np.nan)):
                self._set_result_card_value(self.peak_result_card, f"{peak_result['cbf_hz']:.3f} Hz", f"{peak_result['n_peaks']} peaks")
            else:
                self._set_result_card_value(self.peak_result_card, "—", peak_result.get("note", "No peak result"))

            self.log("\nROI frequency analysis:")
            self.log(f"  Method: {frequency_result['method']}")
            self.log(f"  FPS: {fps:.3f}")
            self.log(f"  Search range: {min_hz:.2f}-{max_hz:.2f} Hz")
            if roi is None:
                self.log("  Region: whole frame (exploratory)")
            else:
                x, y, w, h = roi
                self.log(f"  ROI: x={x}, y={y}, width={w}, height={h}")
            if background_roi is not None:
                bx, by, bw, bh = background_roi
                self.log(f"  Background ROI: x={bx}, y={by}, width={bw}, height={bh}")
            self.log(f"  Frequency CBF: {frequency_result['cbf_hz']:.3f} Hz")
            self.log(f"  Peak/background strength: {frequency_result['peak_to_background']:.3f}")
            if np.isfinite(peak_result.get("cbf_hz", np.nan)):
                self.log(f"  Peak-interval CBF: {peak_result['cbf_hz']:.3f} Hz ({peak_result['n_peaks']} peaks)")
            else:
                self.log(f"  Peak-interval CBF: not available ({peak_result.get('note', 'unknown')})")
            self._plot_frequency_result(signal, fps, frequency_result, peak_result)
        except Exception as exc:
            self.log("\nROI frequency analysis failed:")
            self.log(str(exc))

    def create_kymograph_layer(self):
        try:
            if self.stack is None:
                raise ValueError("No AVI stack loaded.")
            roi = self._get_selected_roi()
            kymo = make_kymograph(self.stack, roi=roi)
            self.viewer.add_image(kymo, name="Cilia kymograph", colormap="gray")
            self.log("\nCreated kymograph layer from selected ROI.")
        except Exception as exc:
            self.log("\nKymograph creation failed:")
            self.log(str(exc))

    def run_cbf_heatmap(self, region_mode: str, method: str, tile_size: int, min_hz: float, max_hz: float):
        try:
            if self.stack is None:
                raise ValueError("No AVI stack loaded.")
            roi = self._roi_for_region_mode(region_mode)
            fps = float(self.fps_box.value())
            result = compute_cbf_heatmap(
                self.stack, fps=fps, min_hz=min_hz, max_hz=max_hz, method=method, tile_size=tile_size, roi=roi
            )
            translate, scale = self._map_layer_placement(roi, result["tile_size"])
            self.viewer.add_image(
                result["cbf_map"],
                name=f"CBF heatmap {method} (Hz)",
                colormap="turbo",
                translate=translate,
                scale=scale,
            )
            self.viewer.add_image(
                result["strength_map"],
                name="CBF peak strength map",
                colormap="magma",
                translate=translate,
                scale=scale,
            )
            self.last_map_result = result
            self.log("\nCBF heatmap analysis:")
            self.log(f"  Region: {region_mode}")
            self.log(f"  Method: {method}")
            self.log(f"  Tile size: {tile_size}")
            finite = np.isfinite(result["cbf_map"])
            if np.any(finite):
                self.log(f"  CBF map median: {float(np.nanmedian(result['cbf_map'])):.3f} Hz")
                self.log(f"  CBF map range: {float(np.nanmin(result['cbf_map'])):.3f}-{float(np.nanmax(result['cbf_map'])):.3f} Hz")
            else:
                self.log("  Warning: no valid CBF pixels were produced.")
            self._plot_map_result(result["cbf_map"], title=f"CBF Heatmap ({method})", colorbar_label="Hz")
        except Exception as exc:
            self.log("\nCBF heatmap analysis failed:")
            self.log(str(exc))

    def run_motion_activity(self, region_mode: str, method: str, tile_size: int, min_hz: float, max_hz: float):
        try:
            if self.stack is None:
                raise ValueError("No AVI stack loaded.")
            roi = self._roi_for_region_mode(region_mode)
            fps = float(self.fps_box.value())
            result = compute_motion_activity_map(
                self.stack,
                method=method,
                fps=fps,
                min_hz=min_hz,
                max_hz=max_hz,
                tile_size=tile_size,
                roi=roi,
            )
            translate, scale = self._map_layer_placement(roi, result["tile_size"])
            self.viewer.add_image(
                result["activity_map"],
                name=f"Motion activity - {result['method']}",
                colormap="viridis",
                translate=translate,
                scale=scale,
            )
            self.last_map_result = result
            self.log("\nMotion activity analysis:")
            self.log(f"  Region: {region_mode}")
            self.log(f"  Method: {result['method']}")
            self.log(f"  Tile size: {tile_size}")
            self._plot_map_result(result["activity_map"], title=result["method"], colorbar_label="activity")
        except Exception as exc:
            self.log("\nMotion activity analysis failed:")
            self.log(str(exc))

    def run_advanced_flow(self, region_mode: str, frame_step: int, max_pairs: int):
        try:
            if self.stack is None:
                raise ValueError("No AVI stack loaded.")
            roi = self._roi_for_region_mode(region_mode)
            result = compute_optical_flow_maps(self.stack, roi=roi, frame_step=frame_step, max_pairs=max_pairs)
            translate, scale = self._map_layer_placement(roi, 1)
            for key, cmap in [
                ("magnitude", "viridis"),
                ("direction", "twilight"),
                ("curl", "coolwarm"),
                ("deformation", "magma"),
            ]:
                self.viewer.add_image(
                    result[key],
                    name=f"Optical flow {key}",
                    colormap=cmap,
                    translate=translate,
                    scale=scale,
                )
            self.last_map_result = result
            self.log("\nAdvanced optical-flow analysis:")
            self.log(f"  Region: {region_mode}")
            self.log(f"  Frame step: {frame_step}")
            self.log(f"  Frame pairs used: {result['n_pairs']}")
            self.log("  Interpretation: exploratory motion-field descriptors, not diagnostic beat-pattern classes.")
            self._plot_map_result(result["magnitude"], title="Optical Flow Magnitude", colorbar_label="pixels/frame")
        except Exception as exc:
            self.log("\nAdvanced optical-flow analysis failed:")
            self.log(str(exc))

    # -------------------------
    # Plotting and export
    # -------------------------
    def _style_plot_axes(self, axes):
        axes.set_facecolor("#070d18")
        axes.tick_params(colors="#d3deec", labelsize=9)
        axes.xaxis.label.set_color("#dbeafe")
        axes.yaxis.label.set_color("#dbeafe")
        axes.title.set_color("#f8fbff")
        axes.title.set_size(11)
        axes.title.set_weight("bold")
        for spine in axes.spines.values():
            spine.set_color("#40546f")
        axes.grid(True, color="#263750", alpha=0.55, linewidth=0.7)

    def _plot_frequency_result(self, signal: np.ndarray, fps: float, frequency_result: dict, peak_result: dict):
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        time = np.arange(signal.size) / fps
        ax1.plot(time, signal, linewidth=1.2)
        peaks = peak_result.get("peaks")
        if peaks is not None and len(peaks) > 0:
            ax1.plot(time[peaks], signal[peaks], "o", markersize=3)
        ax1.set_title("ROI intensity signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Intensity")
        self._style_plot_axes(ax1)

        freqs = frequency_result["freqs"]
        power = frequency_result["power"]
        ax2.plot(freqs, power, linewidth=1.2)
        ax2.axvline(frequency_result["cbf_hz"], linestyle="--", linewidth=1.2)
        ax2.set_xlim(0, max(1, min(frequency_result["effective_max_hz"] * 1.4, frequency_result["nyquist_hz"])))
        ax2.set_title(f"Spectrum; CBF = {frequency_result['cbf_hz']:.3f} Hz")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power")
        self._style_plot_axes(ax2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_map_result(self, data: np.ndarray, title: str, colorbar_label: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(data, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self._style_plot_axes(ax)
        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)
        cbar.ax.yaxis.label.set_color("#dbeafe")
        cbar.ax.tick_params(colors="#d3deec")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def copy_measurement_graph(self):
        self.canvas.draw()
        width, height = self.canvas.get_width_height()
        buf = self.canvas.buffer_rgba()
        from qtpy.QtGui import QImage, QPixmap

        image = QImage(buf, width, height, QImage.Format_RGBA8888).copy()
        QApplication.clipboard().setPixmap(QPixmap.fromImage(image))
        self.log("Copied current graph to clipboard.")

    def save_measurement_graph_as_tiff(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save graph as TIFF", "cilia_analysis_graph.tif", "TIFF files (*.tif *.tiff)")
        if not path:
            return
        self.figure.savefig(path, dpi=300, facecolor=self.figure.get_facecolor())
        self.log(f"Saved graph: {path}")

    def export_last_analysis(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not directory:
            return
        out_dir = Path(directory)
        try:
            if self.last_signal is not None:
                fps = float(self.fps_box.value())
                time = np.arange(self.last_signal.size) / fps
                table = np.column_stack([time, self.last_signal])
                np.savetxt(out_dir / "roi_signal.csv", table, delimiter=",", header="time_sec,signal", comments="")
            if self.last_frequency_result is not None:
                spec = np.column_stack([self.last_frequency_result["freqs"], self.last_frequency_result["power"]])
                np.savetxt(out_dir / "frequency_spectrum.csv", spec, delimiter=",", header="frequency_hz,power", comments="")
            if self.last_map_result is not None:
                for key, value in self.last_map_result.items():
                    if isinstance(value, np.ndarray):
                        np.save(out_dir / f"{key}.npy", value)
            (out_dir / "cilia_assistant_log.txt").write_text(self._log_export_text(), encoding="utf-8")
            self.log(f"Exported last analysis to: {out_dir}")
        except Exception as exc:
            self.log("Export failed:")
            self.log(str(exc))

    # -------------------------
    # Theme
    # -------------------------
    def _apply_ux_theme(self):
        self.setStyleSheet(
            """
            QWidget#ciliaAssistant { background: #080f1c; color: #e8eef7; font-size: 13px; }
            QFrame#assistantHeader { background: #101a2b; border: 1px solid #24344f; border-radius: 8px; }
            QLabel#appLogo { background: #13294b; color: #70e1ff; border: 1px solid #2563eb; border-radius: 8px; min-width: 48px; max-width: 48px; min-height: 48px; max-height: 48px; font-size: 15px; font-weight: 900; }
            QLabel#appTitle { color: #f8fbff; font-size: 22px; font-weight: 850; }
            QLabel#appSubtitle { color: #c3d0e2; font-size: 13px; }
            QScrollArea#assistantScrollArea, QWidget#assistantScrollContent { background: transparent; }
            QFrame#collapsibleBody { background: #101827; border: 1px solid #263750; border-left: 4px solid #38bdf8; border-radius: 8px; }
            QToolButton#collapsibleToggle { background: #0b1321; color: #dcecff; border: 1px solid #7aa7cf; border-radius: 9px; min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; font-weight: 700; padding: 0px; }
            QLabel#collapsibleStepBadge { background: #233a5c; color: #eef7ff; border: 1px solid #5f91bc; border-radius: 9px; padding: 3px 8px; font-size: 12px; font-weight: 800; }
            QLabel#collapsibleTitle { color: #f4f8ff; font-size: 13px; font-weight: 800; padding: 2px 2px; }
            QFrame#collapsibleHeaderLine { color: #263750; }
            QTabWidget::pane { border: 1px solid #263750; border-radius: 6px; background: #0b1321; }
            QTabBar::tab { background: #172236; color: #dbeafe; padding: 7px 8px; border: 1px solid #334967; border-bottom: none; }
            QTabBar::tab:selected { background: #2f6df6; color: white; }
            QPushButton { min-height: 32px; border-radius: 7px; padding: 6px 12px; font-weight: 750; }
            QPushButton[variant="primary"] { background: #2f6df6; border: 1px solid #4d8dff; color: #ffffff; }
            QPushButton[variant="primary"]:hover { background: #3b82f6; border-color: #89b8ff; }
            QPushButton[variant="secondary"] { background: #172236; border: 1px solid #334967; color: #e8eef7; }
            QPushButton[variant="secondary"]:hover { background: #20324e; border-color: #5b789d; }
            QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit { background: #070d18; border: 1px solid #263750; border-radius: 7px; color: #e8eef7; selection-background-color: #2f6df6; selection-color: #ffffff; min-height: 28px; padding: 3px 8px; }
            QCheckBox { color: #cbd7e8; spacing: 8px; font-weight: 650; }
            QTextEdit#runLog { min-height: 120px; padding: 8px; font-family: monospace; color: #dbeafe; }
            QFrame#resultCard { background: #0b1321; border: 1px solid #2b4464; border-radius: 8px; }
            QLabel#resultLabel { color: #b8c7da; font-size: 12px; font-weight: 750; }
            QLabel#resultValue { color: #7cff8a; font-size: 24px; font-weight: 900; padding: 2px 0px; }
            QLabel#resultNote { color: #8fa2bb; font-size: 11px; }
            """
        )
