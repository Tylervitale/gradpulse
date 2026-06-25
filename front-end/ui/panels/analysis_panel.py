import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QFileDialog, QDoubleSpinBox, QSplitter
)
from PyQt6.QtCore import Qt

from ui.components.mpl_widget import MatplotlibWidget
from core.worker import Worker
from gradpulse import validate, liouville_f_proc, liouville_cr_f_proc, liouville_nqubit_closed_f_proc
from gradpulse.profiles import ParametricCouplerProfile, CrossResonanceProfile
from gradpulse.analysis import ParametricCZAnalysisMixin
import numpy as np
from gradpulse.viz import plot_error_budget, plot_robustness

class AnalysisPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()

        self.pulse_data = None

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 1. Top Bar: Load Pulse
        top_bar = QHBoxLayout()
        self.load_btn = QPushButton("Load Pulse JSON")
        self.load_btn.clicked.connect(self.load_pulse)
        self.file_label = QLabel("No file loaded")

        top_bar.addWidget(self.load_btn)
        top_bar.addWidget(self.file_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Splitter for Controls and Views
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 2. Controls Panel (Left)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Analysis Actions
        actions_group = QGroupBox("Analysis Actions")
        actions_layout = QVBoxLayout()

        self.validate_btn = QPushButton("Run QuTiP Cross-Check")
        self.validate_btn.clicked.connect(self.run_validation)
        self.validate_btn.setEnabled(False)
        actions_layout.addWidget(self.validate_btn)

        self.liouville_btn = QPushButton("Run Liouville Check")
        self.liouville_btn.clicked.connect(self.run_liouville)
        self.liouville_btn.setEnabled(False)
        actions_layout.addWidget(self.liouville_btn)

        actions_group.setLayout(actions_layout)
        control_layout.addWidget(actions_group)

        # Plot options
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout()
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(["Error Budget", "Robustness Sweep", "Pulse Spectrogram"])
        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.generate_plot)
        self.plot_btn.setEnabled(False)
        plot_layout.addWidget(self.plot_combo)
        plot_layout.addWidget(self.plot_btn)
        plot_group.setLayout(plot_layout)
        control_layout.addWidget(plot_group)

        # 3. Main Viewer (Right)
        self.plot_widget = MatplotlibWidget()

        splitter.addWidget(control_widget)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

    def load_pulse(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Pulse JSON", "", "JSON Files (*.json)")
        if file_name:
            self.file_label.setText(os.path.basename(file_name))
            self.pulse_file = file_name
            with open(file_name, 'r') as f:
                self.pulse_data = json.load(f)

            # Enable actions
            self.validate_btn.setEnabled(True)
            self.liouville_btn.setEnabled(True)
            self.plot_btn.setEnabled(True)

    def run_validation(self):
        self.validate_btn.setEnabled(False)
        self.validate_btn.setText("Running...")

        def val_task():
            return validate.cross_check(self.pulse_data)

        worker = Worker(val_task)
        worker.signals.result.connect(self.on_val_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.validate_btn, "Run QuTiP Cross-Check"))

        self.threadpool.start(worker)

    def run_liouville(self):
        self.liouville_btn.setEnabled(False)
        self.liouville_btn.setText("Running...")

        def liouville_task():
            if not self.pulse_data:
                raise ValueError("No pulse loaded.")
            arch = self.pulse_data.get('architecture', 'parametric_cz')
            dt_ns = self.pulse_data.get('dt_ns', 1.0)
            target = self.pulse_data.get('target_gate', 'cz')
            waveform = np.array(self.pulse_data.get('waveform', []))
            if arch == 'parametric_cz':
                profile = ParametricCouplerProfile() # mock
                return liouville_f_proc(profile, waveform, target, dt_ns)
            elif arch == 'cross_resonance':
                profile = CrossResonanceProfile() # mock
                return liouville_cr_f_proc(profile, waveform, dt_ns=dt_ns)
            else:
                raise NotImplementedError(f"Liouville check not implemented for architecture: {arch}")

        worker = Worker(liouville_task)
        worker.signals.result.connect(self.on_liouville_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.liouville_btn, "Run Liouville Check"))

        self.threadpool.start(worker)

    def generate_plot(self):
        self.plot_btn.setEnabled(False)
        self.plot_btn.setText("Generating...")

        plot_type = self.plot_combo.currentText()

        def plot_task():
            opt_panel = self.main_window.opt_panel
            if not opt_panel.result or 'best_waveform' not in opt_panel.result:
                raise ValueError("No active optimization result found. Please run an optimization first.")
            result = opt_panel.result

            # Since the API says we need the optimizer to generate the budget, we look it up or mock it.
            # In our current setup, we need the optimizer returned from the `optimize_cz` etc.
            # By default it is returned in result['optimizer']
            if 'optimizer' not in result:
                raise ValueError("Optimization result missing 'optimizer' key.")

            optimizer = result['optimizer']
            raw_param = result.get('best_raw_param', None)

            if plot_type == "Error Budget":
                return "budget", optimizer.error_budget(raw_param)
            elif plot_type == "Robustness Sweep":
                return "robustness", optimizer.robustness_sweep(raw_param)
            elif plot_type == "Pulse Spectrogram":
                from gradpulse.viz import plot_spectrogram
                return "spectrogram", result['best_waveform']

        worker = Worker(plot_task)
        worker.signals.result.connect(self.on_plot_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.plot_btn, "Generate Plot"))

        self.threadpool.start(worker)

    def on_val_success(self, result):
        print(f"QuTiP Validation Result: Gap = {result.get('delta', result.get('gap', 'N/A'))}")

    def on_liouville_success(self, result):
        print(f"Liouville check result: F_proc = {result}")

    def on_plot_success(self, result_tuple):
        ptype, data = result_tuple

        self.plot_widget.clear()

        if ptype == "budget":
            plot_error_budget(data, ax=self.plot_widget.get_axes())
        elif ptype == "robustness":
            # plot_robustness returns a Figure
            fig = plot_robustness(data)
            # Reattach to our canvas
            self.plot_widget.canvas.fig = fig
            self.plot_widget.canvas.draw()
        elif ptype == "spectrogram":
            from gradpulse.viz import plot_spectrogram
            plot_spectrogram(data, ax=self.plot_widget.get_axes())

        self.plot_widget.canvas.draw()

    def on_error(self, error):
        print(f"Error during analysis task: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
