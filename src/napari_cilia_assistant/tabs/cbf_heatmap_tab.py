from __future__ import annotations

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox


class CbfHeatmapTab(QWidget):
    """Spatial CBF heatmap tab."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        note = QLabel(
            "Spatial screen: estimate the dominant frequency across the selected ROI or full frame. "
            "Use this to find heterogeneous beating regions before choosing final ROIs."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.region_combo = QComboBox()
        self.region_combo.addItems(["Selected ROI", "Whole frame"])
        layout.addWidget(self._row("Input region", self.region_combo))

        self.method_combo = QComboBox()
        self.method_combo.addItems(["FFT", "Welch", "Periodogram"])
        layout.addWidget(self._row("Frequency method", self.method_combo))

        self.tile_size_box = QSpinBox()
        self.tile_size_box.setRange(1, 256)
        self.tile_size_box.setValue(8)
        self.tile_size_box.setToolTip("Larger tile size is faster and less noisy. Use 8 or 16 first.")
        layout.addWidget(self._row("Tile size", self.tile_size_box))

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

        self.run_button = QPushButton("Generate CBF Heatmap")
        self.run_button.setToolTip("Adds CBF heatmap and peak-strength map as napari image layers.")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)

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

    def _run(self):
        self.controller.run_cbf_heatmap(
            region_mode=self.region_combo.currentText(),
            method=self.method_combo.currentText(),
            tile_size=int(self.tile_size_box.value()),
            min_hz=float(self.min_hz_box.value()),
            max_hz=float(self.max_hz_box.value()),
        )
