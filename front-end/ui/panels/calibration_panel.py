from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QSpinBox,
    QPushButton, QTextEdit, QSplitter, QComboBox
)
from PyQt6.QtCore import Qt

from core.worker import Worker
from gradpulse.hardware import SimulatedBackend, calibrate_to_hardware
from gradpulse.profiles import ParametricCouplerProfile

class CalibrationPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        calib_group = QGroupBox("Closed-Loop Hardware Calibration")
        calib_form = QFormLayout()

        self.calib_rounds = QSpinBox()
        self.calib_rounds.setRange(1, 20)
        self.calib_rounds.setValue(3)
        calib_form.addRow("Rounds:", self.calib_rounds)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["SimulatedBackend", "QuTiPDeviceBackend"])
        calib_form.addRow("Backend:", self.backend_combo)

        self.calib_btn = QPushButton("Run Hardware Calibration Loop")
        self.calib_btn.clicked.connect(self.run_calibration)
        calib_form.addRow("", self.calib_btn)

        calib_group.setLayout(calib_form)
        control_layout.addWidget(calib_group)

        self.calib_output = QTextEdit()
        self.calib_output.setReadOnly(True)
        self.calib_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; font-size: 13px;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.calib_output)
        splitter.setSizes([350, 650])

        layout.addWidget(splitter)

    def run_calibration(self):
        self.calib_btn.setEnabled(False)
        self.calib_btn.setText("Running...")
        self.calib_output.clear()

        rounds = self.calib_rounds.value()

        backend_type = self.backend_combo.currentText()

        def calib_task():
            # Use active profile from main_window
            if self.main_window.active_profile:
                initial_profile = self.main_window.active_profile
            else:
                initial_profile = ParametricCouplerProfile()

            # The SimulatedBackend emulates hardware behavior.
            if backend_type == "QuTiPDeviceBackend":
                from gradpulse.hardware import QuTiPDeviceBackend
                backend = QuTiPDeviceBackend(initial_profile)
            else:
                backend = SimulatedBackend(initial_profile)

            # Use small number of iterations for the GUI to remain somewhat responsive
            opt_kwargs = {"n_seeds": 1, "iterations": 20, "n_slices": 100}

            return calibrate_to_hardware(initial_profile, backend, rounds=rounds, opt_kwargs=opt_kwargs)

        worker = Worker(calib_task)
        worker.signals.result.connect(self.on_calib_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.calib_btn, "Run Hardware Calibration Loop"))

        self.threadpool.start(worker)

    def on_calib_success(self, calib_res):
        history = calib_res.get('history', [])
        refined_prof = calib_res.get('refined_profile', None)

        text = "--- Calibration Loop Results ---\n\n"
        for entry in history:
            text += f"Round {entry.get('round')}:\n"
            text += f"  Model F_avg:    {entry.get('f_model_avg', 0):.5f}\n"
            text += f"  Hardware F_avg: {entry.get('f_hardware_avg', 0):.5f}\n"
            text += f"  Gap:            {entry.get('gap', 0):.5e}\n"
            text += f"  Coherence Scale:{entry.get('coherence_scale', 0):.5f}\n\n"

        if refined_prof:
            text += "--- Final Refined Profile ---\n"

            # Print relevant values depending on profile type
            if hasattr(refined_prof, 't1_ns_q1'):
                text += f"T1 Q1: {refined_prof.t1_ns_q1} ns\n"
                text += f"T2 Q1: {refined_prof.t2_ns_q1} ns\n"
            elif hasattr(refined_prof, 't1_ns_control'):
                text += f"T1 Control: {refined_prof.t1_ns_control} ns\n"
                text += f"T2 Control: {refined_prof.t2_ns_control} ns\n"
            elif hasattr(refined_prof, 't1_ns'):
                text += f"T1 Q0: {refined_prof.t1_ns[0]} ns\n"
                text += f"T2 Q0: {refined_prof.t2_ns[0]} ns\n"

        self.calib_output.setPlainText(text)

    def on_error(self, error):
        print(f"Calibration Error: {error[1]}")
        self.calib_output.setPlainText(f"Error: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
