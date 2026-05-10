from __future__ import annotations

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox


class RoiFrequencyTab(QWidget):
    """Standard ROI-based CBF measurement tab."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        note = QLabel(
            "Standard measurement: select a cilia ROI, then estimate CBF from the ROI intensity rhythm."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["FFT", "Welch", "Periodogram"])
        self.method_combo.setToolTip("FFT is simple and transparent. Welch is more stable for noisy traces.")
        layout.addWidget(self._row("Frequency method", self.method_combo))

        self.min_hz_box = QDoubleSpinBox()
        self.min_hz_box.setRange(0.1, 500)
        self.min_hz_box.setDecimals(2)
        self.min_hz_box.setValue(3.0)
        layout.addWidget(self._row("Min Hz", self.min_hz_box))

        self.max_hz_box = QDoubleSpinBox()
        self.max_hz_box.setRange(0.1, 500)
        self.max_hz_box.setDecimals(2)
        self.max_hz_box.setValue(25.0)
        layout.addWidget(self._row("Max Hz", self.max_hz_box))

        self.measure_roi_button = QPushButton("Analyze Selected ROI")
        self.measure_roi_button.setToolTip("Primary workflow. Use this for publication-style ROI CBF measurement.")
        self.measure_roi_button.clicked.connect(self._measure_roi)
        layout.addWidget(self.measure_roi_button)

        self.measure_whole_button = QPushButton("Analyze Whole Frame")
        self.measure_whole_button.setToolTip("Exploratory only. Whole-frame signals may include tissue drift or illumination changes.")
        self.measure_whole_button.clicked.connect(self._measure_whole)
        layout.addWidget(self.measure_whole_button)

        self.kymo_button = QPushButton("Create Kymograph from Selected ROI")
        self.kymo_button.setToolTip("Create a time-vs-position audit layer for visual checking of periodic motion.")
        self.kymo_button.clicked.connect(self._make_kymograph)
        layout.addWidget(self.kymo_button)

        layout.addStretch(1)
        self.setLayout(layout)

    def _row(self, label: str, widget: QWidget) -> QWidget:
        box = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lab = QLabel(label)
        lab.setMinimumWidth(120)
        row.addWidget(lab)
        row.addWidget(widget, 1)
        box.setLayout(row)
        return box

    def _params(self):
        return self.method_combo.currentText(), float(self.min_hz_box.value()), float(self.max_hz_box.value())

    def _measure_roi(self):
        method, min_hz, max_hz = self._params()
        self.controller.run_roi_frequency(method=method, min_hz=min_hz, max_hz=max_hz, whole_frame=False)

    def _measure_whole(self):
        method, min_hz, max_hz = self._params()
        self.controller.run_roi_frequency(method=method, min_hz=min_hz, max_hz=max_hz, whole_frame=True)

    def _make_kymograph(self):
        self.controller.create_kymograph_layer()
