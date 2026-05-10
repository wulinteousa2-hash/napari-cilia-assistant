from __future__ import annotations

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox


class AdvancedFlowTab(QWidget):
    """Experimental optical-flow and pattern descriptor tab."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        note = QLabel(
            "Advanced / experimental: Farneback optical flow estimates apparent image motion. "
            "Treat magnitude, direction, curl, and deformation as exploratory descriptors, not diagnostic classifications."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.region_combo = QComboBox()
        self.region_combo.addItems(["Selected ROI", "Whole frame"])
        layout.addWidget(self._row("Input region", self.region_combo))

        self.frame_step_box = QSpinBox()
        self.frame_step_box.setRange(1, 50)
        self.frame_step_box.setValue(1)
        layout.addWidget(self._row("Frame step", self.frame_step_box))

        self.max_pairs_box = QSpinBox()
        self.max_pairs_box.setRange(1, 5000)
        self.max_pairs_box.setValue(100)
        self.max_pairs_box.setToolTip("Limits runtime for long videos. Start with 100.")
        layout.addWidget(self._row("Max frame pairs", self.max_pairs_box))

        self.run_button = QPushButton("Run Optical Flow Analysis")
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
        self.controller.run_advanced_flow(
            region_mode=self.region_combo.currentText(),
            frame_step=int(self.frame_step_box.value()),
            max_pairs=int(self.max_pairs_box.value()),
        )
