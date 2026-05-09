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
    QCheckBox,
    QHBoxLayout,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QToolButton,
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


class CollapsiblePanel(QWidget):
    """Compact section panel matching the other napari assistant tools."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        collapsed: bool = False,
        parent: QWidget | None = None,
    ) -> None:
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

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._on_toggled(collapsed)

    def _split_title(self, title: str) -> tuple[str, str]:
        normalized = title.strip()
        head, separator, tail = normalized.partition(".")
        if separator and head.strip() and tail.strip():
            return head.strip(), tail.strip()
        return "", normalized

    def _on_toggled(self, checked: bool) -> None:
        self.toggle_button.setText("+" if checked else "−")
        self.body_frame.setVisible(not checked)
        self.updateGeometry()


class CiliaAssistantWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.setObjectName("ciliaAssistant")
        self.setMinimumWidth(360)
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
        self.last_fft_result: dict | None = None
        self.last_peak_result: dict | None = None

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header_frame = QFrame()
        header_frame.setObjectName("assistantHeader")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(12)

        self.logo = QLabel("CBF")
        self.logo.setObjectName("appLogo")
        self.logo.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.logo)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(2)

        self.title = QLabel("Cilia Assistant")
        self.title.setObjectName("appTitle")
        title_column.addWidget(self.title)

        self.subtitle = QLabel("ROI-based beat-frequency measurement for cilia videos")
        self.subtitle.setObjectName("appSubtitle")
        self.subtitle.setWordWrap(True)
        title_column.addWidget(self.subtitle)

        header_layout.addLayout(title_column, 1)
        header_frame.setLayout(header_layout)
        layout.addWidget(header_frame)

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

        # -------------------------
        # Results section
        # -------------------------
        results_box = QWidget()
        results_layout = QHBoxLayout()
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        self.fft_result_card = self._build_result_card("FFT CBF", "—", "Primary estimate")
        self.peak_result_card = self._build_result_card("Peak CBF", "—", "Independent check")
        results_layout.addWidget(self.fft_result_card)
        results_layout.addWidget(self.peak_result_card)
        results_box.setLayout(results_layout)

        content_layout.addWidget(CollapsiblePanel("Results. Key Measurements", results_box, collapsed=False))

        # -------------------------
        # Input section
        # -------------------------
        input_box = QWidget()
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
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
        content_layout.addWidget(CollapsiblePanel("Step 1. Input", input_box, collapsed=False))

        # -------------------------
        # ROI section
        # -------------------------
        roi_box = QWidget()
        roi_layout = QVBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(8)

        self.create_roi_button = QPushButton("Create / Edit ROI Rectangle")
        self._style_button(self.create_roi_button, "secondary")
        self.create_roi_button.setToolTip(
            "Create a rectangle over an active ciliated edge; ROI placement is the main user-controlled scientific decision."
        )
        self.create_roi_button.clicked.connect(self.create_or_select_roi_layer)
        roi_layout.addWidget(self.create_roi_button)

        self.create_background_button = QPushButton("Create / Edit Background ROI")
        self._style_button(self.create_background_button, "secondary")
        self.create_background_button.setToolTip(
            "Optional negative-control rectangle in a nearby non-cilia region. "
            "Use it to subtract shared illumination or focus drift from the cilia ROI signal."
        )
        self.create_background_button.clicked.connect(self.create_or_select_background_roi_layer)
        roi_layout.addWidget(self.create_background_button)

        self.subtract_background_check = QCheckBox("Subtract background ROI when measuring")
        self.subtract_background_check.setChecked(True)
        self.subtract_background_check.setToolTip(
            "When a background ROI exists, subtract its mean-intensity trace from the selected cilia ROI before FFT."
        )
        roi_layout.addWidget(self.subtract_background_check)

        self.measure_roi_button = QPushButton("Analyze Selected ROI")
        self._style_button(self.measure_roi_button, "primary")
        self.measure_roi_button.setToolTip(
            "Measure ROI mean-intensity oscillation, then estimate CBF using FFT and peak intervals."
        )
        self.measure_roi_button.clicked.connect(self.measure_selected_roi)
        roi_layout.addWidget(self.measure_roi_button)

        self.measure_whole_button = QPushButton("Analyze Whole-Frame Motion")
        self._style_button(self.measure_whole_button, "secondary")
        self.measure_whole_button.setToolTip(
            "Exploratory only. Whole-frame signals may include illumination drift, stage motion, or tissue motion."
        )
        self.measure_whole_button.clicked.connect(self.measure_whole_frame)
        roi_layout.addWidget(self.measure_whole_button)

        roi_box.setLayout(roi_layout)
        content_layout.addWidget(CollapsiblePanel("Step 2. Region of Interest", roi_box, collapsed=False))

        # -------------------------
        # Analysis parameters
        # -------------------------
        analysis_box = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_layout.setContentsMargins(0, 0, 0, 0)
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
        content_layout.addWidget(CollapsiblePanel("Step 3. Frequency Analysis", analysis_box, collapsed=False))

        # -------------------------
        # Plot section
        # -------------------------
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
        plot_panel = CollapsiblePanel("Step 4. Measurement Graph", plot_box, collapsed=False)
        plot_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(plot_panel, 1)

        # -------------------------
        # Log section
        # -------------------------
        log_box = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
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
        self.output.setMinimumHeight(140)
        self.output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.output)

        log_box.setLayout(log_layout)
        log_panel = CollapsiblePanel("Step 6. Log", log_box, collapsed=True)
        log_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(log_panel, 1)

        scroll_content.setLayout(content_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)

        self.setLayout(layout)
        self._apply_ux_theme()

    def _style_button(self, button: QPushButton, variant: str):
        button.setProperty("variant", variant)
        button.setCursor(Qt.PointingHandCursor)

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
        value_label.setProperty("role", label.lower().replace(" ", "_"))
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

    def _apply_ux_theme(self):
        self.setStyleSheet(
            """
            QWidget#ciliaAssistant {
                background: #080f1c;
                color: #e8eef7;
                font-size: 13px;
            }

            QFrame#assistantHeader {
                background: #101a2b;
                border: 1px solid #24344f;
                border-radius: 8px;
            }

            QLabel#appLogo {
                background: #13294b;
                color: #70e1ff;
                border: 1px solid #2563eb;
                border-radius: 8px;
                min-width: 48px;
                max-width: 48px;
                min-height: 48px;
                max-height: 48px;
                font-size: 15px;
                font-weight: 900;
            }

            QLabel#appTitle {
                color: #f8fbff;
                font-size: 22px;
                font-weight: 850;
                padding: 0px;
            }

            QLabel#appSubtitle {
                color: #c3d0e2;
                font-size: 13px;
                padding: 0px;
            }

            QScrollArea#assistantScrollArea,
            QWidget#assistantScrollContent {
                background: transparent;
            }

            QFrame#collapsibleBody {
                background: #101827;
                border: 1px solid #263750;
                border-left: 4px solid #38bdf8;
                border-radius: 8px;
            }

            QToolButton#collapsibleToggle {
                background: #0b1321;
                color: #dcecff;
                border: 1px solid #7aa7cf;
                border-radius: 9px;
                min-width: 22px;
                max-width: 22px;
                min-height: 22px;
                max-height: 22px;
                font-weight: 700;
                padding: 0px;
            }

            QToolButton#collapsibleToggle:hover {
                background: #13243a;
                border-color: #a7d7ff;
                color: #ffffff;
            }

            QLabel#collapsibleStepBadge {
                background: #233a5c;
                color: #eef7ff;
                border: 1px solid #5f91bc;
                border-radius: 9px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: 800;
            }

            QLabel#collapsibleTitle {
                color: #f4f8ff;
                font-size: 13px;
                font-weight: 800;
                padding: 2px 2px;
            }

            QFrame#collapsibleHeaderLine {
                color: #263750;
            }

            QPushButton {
                min-height: 32px;
                border-radius: 7px;
                padding: 6px 12px;
                font-weight: 750;
            }

            QPushButton[variant="primary"] {
                background: #2f6df6;
                border: 1px solid #4d8dff;
                color: #ffffff;
            }

            QPushButton[variant="primary"]:hover {
                background: #3b82f6;
                border-color: #89b8ff;
            }

            QPushButton[variant="secondary"] {
                background: #172236;
                border: 1px solid #334967;
                color: #e8eef7;
            }

            QPushButton[variant="secondary"]:hover {
                background: #20324e;
                border-color: #5b789d;
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
                background: #070d18;
                border: 1px solid #263750;
                border-radius: 7px;
                color: #e8eef7;
                selection-background-color: #2f6df6;
                selection-color: #ffffff;
            }

            QCheckBox {
                color: #cbd7e8;
                spacing: 8px;
                font-weight: 650;
            }

            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }

            QCheckBox::indicator:unchecked {
                background: #070d18;
                border: 1px solid #45637f;
                border-radius: 3px;
            }

            QCheckBox::indicator:checked {
                background: #2f6df6;
                border: 1px solid #70e1ff;
                border-radius: 3px;
            }

            QSpinBox,
            QDoubleSpinBox {
                min-height: 28px;
                padding: 3px 8px;
            }

            QSpinBox:focus,
            QDoubleSpinBox:focus,
            QTextEdit:focus {
                border-color: #70e1ff;
            }

            QTextEdit#runLog {
                min-height: 120px;
                padding: 8px;
                font-family: monospace;
                color: #dbeafe;
            }

            QFrame#resultCard {
                background: #0b1321;
                border: 1px solid #2b4464;
                border-radius: 8px;
            }

            QLabel#resultLabel {
                color: #b8c7da;
                font-size: 12px;
                font-weight: 750;
            }

            QLabel#resultValue {
                color: #7cff8a;
                font-size: 24px;
                font-weight: 900;
                padding: 2px 0px;
            }

            QLabel#resultNote {
                color: #8fa2bb;
                font-size: 11px;
            }

            QScrollBar:vertical {
                background: #0b1321;
                width: 12px;
                margin: 2px;
            }

            QScrollBar::handle:vertical {
                background: #45637f;
                border-radius: 5px;
                min-height: 28px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def _style_plot_axes(self, axes):
        axes.set_facecolor("#070d18")
        axes.tick_params(colors="#d3deec", labelsize=9)
        axes.xaxis.label.set_color("#dbeafe")
        axes.yaxis.label.set_color("#dbeafe")
        axes.title.set_color("#f8fbff")
        axes.xaxis.label.set_size(9)
        axes.yaxis.label.set_size(9)
        axes.title.set_size(11)
        for spine in axes.spines.values():
            spine.set_color("#40546f")
        axes.grid(True, color="#263750", alpha=0.55, linewidth=0.7)

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
        return self._find_layer(self.roi_layer_name)

    def _find_background_roi_layer(self):
        return self._find_layer(self.background_roi_layer_name)

    def _find_layer(self, layer_name: str):
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def create_or_select_roi_layer(self):
        self._create_or_select_rectangle_layer(
            layer_name=self.roi_layer_name,
            log_name="ROI",
            edge_color="yellow",
            face_color=[1, 1, 0, 0.10],
            y_bounds=(0.40, 0.60),
            x_bounds=(0.40, 0.60),
        )

    def create_or_select_background_roi_layer(self):
        self._create_or_select_rectangle_layer(
            layer_name=self.background_roi_layer_name,
            log_name="background ROI",
            edge_color="cyan",
            face_color=[0, 1, 1, 0.08],
            y_bounds=(0.08, 0.22),
            x_bounds=(0.40, 0.60),
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

        y0 = int(y_size * y_bounds[0])
        y1 = int(y_size * y_bounds[1])
        x0 = int(x_size * x_bounds[0])
        x1 = int(x_size * x_bounds[1])

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
            name=layer_name,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=2,
        )

        roi_layer.mode = "select"
        self.viewer.layers.selection.active = roi_layer

        self.log(f"Created {log_name} rectangle. Move/resize it before measuring.")

    def _get_selected_roi(self) -> tuple[int, int, int, int]:
        return self._get_roi_from_layer(
            layer_name=self.roi_layer_name,
            missing_message="No ROI layer found. Click 'Create / Edit ROI Rectangle' first.",
        )

    def _get_background_roi(self) -> tuple[int, int, int, int]:
        return self._get_roi_from_layer(
            layer_name=self.background_roi_layer_name,
            missing_message="No background ROI layer found.",
        )

    def _get_roi_from_layer(
        self,
        layer_name: str,
        missing_message: str,
    ) -> tuple[int, int, int, int]:
        if self.stack is None:
            raise ValueError("No AVI stack loaded.")

        roi_layer = self._find_layer(layer_name)

        if roi_layer is None:
            raise ValueError(missing_message)

        if len(roi_layer.data) == 0:
            raise ValueError(f"{layer_name} layer exists but contains no shape.")

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

        raw_signal = roi_mean_signal(self.stack, roi=roi)
        background_roi = None
        background_signal = None
        signal = raw_signal

        if roi is not None and self.subtract_background_check.isChecked():
            background_layer = self._find_background_roi_layer()
            if background_layer is not None and len(background_layer.data) > 0:
                background_roi = self._get_background_roi()
                background_signal = roi_mean_signal(self.stack, roi=background_roi)
                signal = raw_signal - background_signal

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
        self.last_background_roi = background_roi
        self.last_signal = signal
        self.last_raw_signal = raw_signal
        self.last_background_signal = background_signal
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
            if background_roi is not None:
                bx, by, bw, bh = background_roi
                self.log(f"  Background ROI: x={bx}, y={by}, width={bw}, height={bh}")
                self.log("  Signal used: ROI mean intensity minus background ROI mean intensity")
            elif self.subtract_background_check.isChecked():
                self.log("  Background correction: not used; no background ROI found.")
            else:
                self.log("  Background correction: off")

        self.log(f"  FPS used: {fps:.3f}")
        self.log(f"  Search range: {min_hz:.2f}–{fft_result['effective_max_hz']:.2f} Hz")
        self.log(f"  Nyquist limit: {fft_result['nyquist_hz']:.2f} Hz")

        self.log(f"  FFT CBF: {fft_result['cbf_hz']:.3f} Hz")
        self.log(f"  FFT peak/background: {fft_result['peak_to_background']:.2f}")
        self._set_result_card_value(
            self.fft_result_card,
            f"{fft_result['cbf_hz']:.3f} Hz",
            f"Peak/background {fft_result['peak_to_background']:.2f}"
            + (" · BG corrected" if background_roi is not None else ""),
        )

        peak_cbf = peak_result.get("cbf_hz", np.nan)
        if np.isfinite(peak_cbf):
            self.log(f"  Peak-interval CBF: {peak_cbf:.3f} Hz")
            self.log(f"  Detected peaks: {peak_result.get('n_peaks', 0)}")
            self._set_result_card_value(
                self.peak_result_card,
                f"{peak_cbf:.3f} Hz",
                f"{peak_result.get('n_peaks', 0)} detected peaks",
            )
        else:
            self.log("  Peak-interval CBF: not reliable")
            self.log(f"  Peak note: {peak_result.get('note', 'N/A')}")
            self._set_result_card_value(
                self.peak_result_card,
                "Not reliable",
                peak_result.get("note", "Check ROI/video"),
            )

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
        signal_label = "background-corrected signal" if self.last_background_signal is not None else "mean intensity"
        ax1.set_title(f"{label}: {signal_label}", loc="left", pad=8)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Intensity")

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
        ax2.text(
            0.98,
            0.90,
            f"FFT CBF\n{cbf_hz:.3f} Hz",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            color="#7cff8a",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#0b1321",
                "edgecolor": "#2b4464",
                "alpha": 0.92,
            },
        )
        ax2.set_xlim(0, min(fft_result["effective_max_hz"] * 1.2, fps / 2.0))
        ax2.set_title("FFT power spectrum", loc="left", pad=8)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power")

        self.figure.subplots_adjust(
            left=0.14,
            right=0.98,
            top=0.94,
            bottom=0.11,
            hspace=0.58,
        )
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

        header = "time_sec,roi_signal_used"
        columns = [time, self.last_signal]
        if self.last_raw_signal is not None and self.last_background_signal is not None:
            columns = [time, self.last_raw_signal, self.last_background_signal, self.last_signal]
            header = (
                "time_sec,roi_mean_intensity,background_mean_intensity,"
                "background_corrected_intensity"
            )

        signal_table = np.column_stack(columns)
        np.savetxt(
            signal_path,
            signal_table,
            delimiter=",",
            header=header,
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
