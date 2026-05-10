from __future__ import annotations

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox


class MotionActivityTab(QWidget):
    """Motion/activity map tab."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        note = QLabel(
            "Motion locator: show where the video changes over time. This is not CBF; use it to find active cilia, drift, or debris."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.region_combo = QComboBox()
        self.region_combo.addItems(["Selected ROI", "Whole frame"])
        layout.addWidget(self._row("Input region", self.region_combo))

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Temporal SD",
            "Frame Difference",
            "Max-Min",
            "Band-limited FFT Power",
        ])
        layout.addWidget(self._row("Activity method", self.method_combo))

        self.tile_size_box = QSpinBox()
        self.tile_size_box.setRange(1, 256)
        self.tile_size_box.setValue(4)
        layout.addWidget(self._row("Tile size", self.tile_size_box))

        self.min_hz_box = QDoubleSpinBox()
        self.min_hz_box.setRange(0.1, 500)
        self.min_hz_box.setDecimals(2)
        self.min_hz_box.setValue(3.0)
        self.min_hz_box.setToolTip("Used only for band-limited FFT power.")
        layout.addWidget(self._row("Min Hz", self.min_hz_box))

        self.max_hz_box = QDoubleSpinBox()
        self.max_hz_box.setRange(0.1, 500)
        self.max_hz_box.setDecimals(2)
        self.max_hz_box.setValue(25.0)
        self.max_hz_box.setToolTip("Used only for band-limited FFT power.")
        layout.addWidget(self._row("Max Hz", self.max_hz_box))

        self.run_button = QPushButton("Generate Motion Activity Map")
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
        self.controller.run_motion_activity(
            region_mode=self.region_combo.currentText(),
            method=self.method_combo.currentText(),
            tile_size=int(self.tile_size_box.value()),
            min_hz=float(self.min_hz_box.value()),
            max_hz=float(self.max_hz_box.value()),
        )
