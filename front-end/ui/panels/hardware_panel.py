from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QSpinBox, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt

from core.worker import Worker
from gradpulse import openpulse_export
from gradpulse.braket_bridge import estimate_experiment_cost
from gradpulse.hardware import SimulatedBackend, calibrate_to_hardware
from gradpulse.profiles import ParametricCouplerProfile

class HardwarePanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Side: Controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # OpenPulse Export
        op_group = QGroupBox("OpenPulse 3.0 Export")
        op_layout = QVBoxLayout()

        self.export_btn = QPushButton("Generate OpenPulse Code")
        self.export_btn.clicked.connect(self.generate_openpulse)
        op_layout.addWidget(self.export_btn)

        op_group.setLayout(op_layout)
        control_layout.addWidget(op_group)

        # Braket Cost Estimation
        braket_group = QGroupBox("Amazon Braket Config")
        braket_form = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Rigetti-Cepheus-1-108Q", "Rigetti-Aspen-M-3"])
        braket_form.addRow("Device:", self.device_combo)

        self.circuits_spin = QSpinBox()
        self.circuits_spin.setRange(1, 1000)
        self.circuits_spin.setValue(100)
        braket_form.addRow("Circuits:", self.circuits_spin)

        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(10, 10000)
        self.shots_spin.setValue(1000)
        braket_form.addRow("Shots:", self.shots_spin)

        self.cost_btn = QPushButton("Estimate Cost")
        self.cost_btn.clicked.connect(self.estimate_cost)
        braket_form.addRow("", self.cost_btn)

        braket_group.setLayout(braket_form)
        control_layout.addWidget(braket_group)


        # Right Side: Text Editor for Generated Code / Output
        self.code_viewer = QTextEdit()
        self.code_viewer.setReadOnly(True)
        self.code_viewer.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; font-size: 13px;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.code_viewer)
        splitter.setSizes([350, 650])

        layout.addWidget(splitter)

    def generate_openpulse(self):
        # We assume optimization_panel contains the result, to demonstrate interconnectivity.
        opt_panel = self.main_window.opt_panel
        if not opt_panel.result or 'best_waveform' not in opt_panel.result:
            self.code_viewer.setPlainText("No valid pulse found. Please run an optimization first in the 'Optimization' tab.")
            return

        pulse = opt_panel.result['best_waveform']

        def export_task():
            # iq_waveform will provide a dict with "iq" mapping to the correct format for to_openpulse_program if use_drag is True
            # to_openpulse_program can also accept simple waveform array
            try:
                # First try to get the full IQ with DRAG if possible
                if 'optimizer' in opt_panel.result:
                    iq = opt_panel.result['optimizer'].iq_waveform(opt_panel.result['best_raw_param'])
                    return openpulse_export.to_openpulse_program(iq)
                else:
                    return openpulse_export.to_openpulse_program(pulse)
            except Exception as e:
                 return f"Error exporting to OpenPulse: {e}\n\nFalling back to simple waveform array...\n" + openpulse_export.to_openpulse_program(pulse)

        worker = Worker(export_task)
        worker.signals.result.connect(self.on_export_success)
        worker.signals.error.connect(self.on_error)

        self.threadpool.start(worker)

    def estimate_cost(self):
        device = self.device_combo.currentText()
        circuits = self.circuits_spin.value()
        shots = self.shots_spin.value()

        def cost_task():
            # In a real app we would pass the actual device arn
            return estimate_experiment_cost(circuits, shots, device="fake-arn")

        worker = Worker(cost_task)
        worker.signals.result.connect(self.on_cost_success)
        worker.signals.error.connect(self.on_error)

        self.threadpool.start(worker)

    def on_export_success(self, code_str):
        self.code_viewer.setPlainText(code_str)
        print("OpenPulse export generated successfully.")

    def on_cost_success(self, cost_estimate):
        # Format the CostEstimate dataclass output
        text = f"Estimated Cost for {cost_estimate.n_circuits} circuits at {cost_estimate.n_shots} shots:\n"
        text += f"Total USD: ${cost_estimate.total_usd:.2f}"
        self.code_viewer.setPlainText(text)
        print(text)

    def on_error(self, error):
        print(f"Error during hardware operation: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
