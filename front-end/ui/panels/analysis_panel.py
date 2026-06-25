import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QFileDialog, QDoubleSpinBox, QSplitter, QTabWidget, QTextEdit, QSpinBox
)
from PyQt6.QtCore import Qt

from ui.components.mpl_widget import MatplotlibWidget
from core.worker import Worker
from gradpulse import validate, liouville_f_proc, liouville_cr_f_proc, liouville_nqubit_closed_f_proc
from gradpulse.profiles import ParametricCouplerProfile
from gradpulse.crossresonance import CrossResonanceProfile
from gradpulse.analysis import ParametricCZAnalysisMixin
import numpy as np
from gradpulse.viz import plot_error_budget, plot_robustness, plot_state_heatmap, plot_bloch_trajectory
from gradpulse.rb import interleaved_rb, native_superops, gate_superoperator
from gradpulse.headtohead import run_head_to_head

class AnalysisPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()

        self.pulse_data = None

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        # 1. Validation & Plotting Tab (Original content)
        self.val_tab = QWidget()
        self.init_val_tab()
        self.tabs.addTab(self.val_tab, "Validation & Plotting")

        # 2. Randomized Benchmarking Tab
        self.rb_tab = QWidget()
        self.init_rb_tab()
        self.tabs.addTab(self.rb_tab, "Randomized Benchmarking")

        # 3. Head-to-Head Tab
        self.h2h_tab = QWidget()
        self.init_h2h_tab()
        self.tabs.addTab(self.h2h_tab, "Head-to-Head")

        # 4. Advanced Diagnostics Tab
        self.adv_diag_tab = QWidget()
        self.init_adv_diag_tab()
        self.tabs.addTab(self.adv_diag_tab, "Advanced Diagnostics")

        layout.addWidget(self.tabs)

    def init_val_tab(self):
        layout = QVBoxLayout(self.val_tab)

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
        self.plot_combo.addItems(["Error Budget", "Robustness Sweep", "Pulse Spectrogram", "State Heatmap", "Bloch Trajectory"])
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

    def init_rb_tab(self):
        layout = QVBoxLayout(self.rb_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        rb_group = QGroupBox("Interleaved RB Settings")
        rb_form = QFormLayout()

        self.rb_sequences = QSpinBox()
        self.rb_sequences.setRange(10, 1000)
        self.rb_sequences.setValue(40)
        rb_form.addRow("Sequences:", self.rb_sequences)

        self.run_rb_btn = QPushButton("Run Simulated IRB")
        self.run_rb_btn.clicked.connect(self.run_rb)
        rb_form.addRow("", self.run_rb_btn)

        rb_group.setLayout(rb_form)
        control_layout.addWidget(rb_group)

        self.rb_output = QTextEdit()
        self.rb_output.setReadOnly(True)
        self.rb_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.rb_output)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

    def init_h2h_tab(self):
        layout = QVBoxLayout(self.h2h_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        h2h_group = QGroupBox("Head-to-Head Config")
        h2h_form = QFormLayout()

        self.h2h_min_dur = QSpinBox()
        self.h2h_min_dur.setRange(10, 500)
        self.h2h_min_dur.setValue(50)
        h2h_form.addRow("Min Duration (ns):", self.h2h_min_dur)

        self.h2h_max_dur = QSpinBox()
        self.h2h_max_dur.setRange(50, 1000)
        self.h2h_max_dur.setValue(200)
        h2h_form.addRow("Max Duration (ns):", self.h2h_max_dur)

        self.h2h_steps = QSpinBox()
        self.h2h_steps.setRange(2, 50)
        self.h2h_steps.setValue(5)
        h2h_form.addRow("Steps:", self.h2h_steps)

        self.run_h2h_btn = QPushButton("Run Head-to-Head Comparison")
        self.run_h2h_btn.clicked.connect(self.run_h2h)
        h2h_form.addRow("", self.run_h2h_btn)

        h2h_group.setLayout(h2h_form)
        control_layout.addWidget(h2h_group)

        self.h2h_output = QTextEdit()
        self.h2h_output.setReadOnly(True)
        self.h2h_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.h2h_output)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

    def init_adv_diag_tab(self):
        layout = QVBoxLayout(self.adv_diag_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        diag_group = QGroupBox("Diagnostic Tool")
        diag_form = QFormLayout()

        self.diag_combo = QComboBox()
        self.diag_combo.addItems(["Quasi-Static Fidelity", "Colored Noise Fidelity", "Spectator Fidelity", "TLS Defect Fidelity"])
        diag_form.addRow("Diagnostic:", self.diag_combo)

        self.run_diag_btn = QPushButton("Run Diagnostic")
        self.run_diag_btn.clicked.connect(self.run_diagnostic)
        diag_form.addRow("", self.run_diag_btn)

        diag_group.setLayout(diag_form)
        control_layout.addWidget(diag_group)

        self.diag_output = QTextEdit()
        self.diag_output.setReadOnly(True)
        self.diag_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.diag_output)
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
            elif plot_type == "State Heatmap":
                # Assuming state heatmap uses result directly or needs a specific eval
                return "state_heatmap", result
            elif plot_type == "Bloch Trajectory":
                return "bloch_trajectory", result

        worker = Worker(plot_task)
        worker.signals.result.connect(self.on_plot_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.plot_btn, "Generate Plot"))

        self.threadpool.start(worker)

    def run_rb(self):
        self.run_rb_btn.setEnabled(False)
        self.run_rb_btn.setText("Running...")
        self.rb_output.clear()

        n_seq = self.rb_sequences.value()

        def rb_task():
            opt_panel = self.main_window.opt_panel
            if not opt_panel.result or 'best_waveform' not in opt_panel.result:
                raise ValueError("No active optimization result found. Please run an optimization first.")
            result = opt_panel.result
            if 'optimizer' not in result:
                raise ValueError("Optimization result missing 'optimizer' key.")

            opt = result['optimizer']
            raw_param = result['best_raw_param']

            # Use gradpulse.rb to calculate the superoperator and run IRB
            sup = gate_superoperator(opt, raw_param)
            rb_res = interleaved_rb(sup, n_sequences=n_seq)
            return rb_res

        worker = Worker(rb_task)
        worker.signals.result.connect(self.on_rb_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_rb_btn, "Run Simulated IRB"))

        self.threadpool.start(worker)

    def run_h2h(self):
        self.run_h2h_btn.setEnabled(False)
        self.run_h2h_btn.setText("Running...")
        self.h2h_output.clear()

        min_d = self.h2h_min_dur.value()
        max_d = self.h2h_max_dur.value()
        steps = self.h2h_steps.value()

        def h2h_task():
            durations = np.linspace(min_d, max_d, steps)

            opt_panel = self.main_window.opt_panel
            if opt_panel.loaded_profile:
                profile = opt_panel.loaded_profile
            else:
                profile = ParametricCouplerProfile() # Fallback

            # iterations kept low for GUI responsiveness
            return run_head_to_head(profile, durations, iterations=50, n_seeds=1, verbose=False)

        worker = Worker(h2h_task)
        worker.signals.result.connect(self.on_h2h_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_h2h_btn, "Run Head-to-Head Comparison"))

        self.threadpool.start(worker)

    def run_diagnostic(self):
        self.run_diag_btn.setEnabled(False)
        self.run_diag_btn.setText("Running...")
        self.diag_output.clear()

        diag_type = self.diag_combo.currentText()

        def diag_task():
            opt_panel = self.main_window.opt_panel
            if not opt_panel.result or 'best_waveform' not in opt_panel.result:
                raise ValueError("No active optimization result found. Please run an optimization first.")
            result = opt_panel.result
            if 'optimizer' not in result:
                raise ValueError("Optimization result missing 'optimizer' key.")

            opt = result['optimizer']
            raw_param = result.get('best_raw_param', None)

            if diag_type == "Quasi-Static Fidelity":
                return diag_type, opt.quasi_static_fidelity(raw_param)
            elif diag_type == "Colored Noise Fidelity":
                return diag_type, opt.colored_noise_fidelity(raw_param)
            elif diag_type == "Spectator Fidelity":
                return diag_type, opt.spectator_fidelity(raw_param)
            elif diag_type == "TLS Defect Fidelity":
                return diag_type, opt.tls_defect_fidelity(raw_param)

        worker = Worker(diag_task)
        worker.signals.result.connect(self.on_diag_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_diag_btn, "Run Diagnostic"))
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
            fig = plot_robustness(data)
            self.plot_widget.canvas.fig = fig
            self.plot_widget.canvas.draw()
        elif ptype == "spectrogram":
            from gradpulse.viz import plot_spectrogram
            plot_spectrogram(data, ax=self.plot_widget.get_axes())
        elif ptype == "state_heatmap":
            plot_state_heatmap(data, ax=self.plot_widget.get_axes())
        elif ptype == "bloch_trajectory":
            plot_bloch_trajectory(data, ax=self.plot_widget.get_axes())

        self.plot_widget.canvas.draw()

    def on_rb_success(self, rb_res):
        text = "--- IRB Results ---\n"
        text += f"Naive CZ Error: {rb_res.get('r_cz_naive', 'N/A')}\n"
        text += f"Leakage-Aware CZ Error: {rb_res.get('r_cz_leakage_aware', 'N/A')}\n"
        text += f"CZ Fidelity (IRB): {rb_res.get('f_cz_irb', 'N/A')}\n"
        text += f"Leakage/Clifford (L1): {rb_res.get('leakage_per_clifford_L1', 'N/A')}\n"
        self.rb_output.setPlainText(text)

    def on_h2h_success(self, h2h_res):
        summary = h2h_res.get('summary', {})
        text = "--- Head-to-Head Summary ---\n"
        text += f"In-Loop Chosen Duration: {summary.get('in_loop_best_duration', 'N/A')} ns\n"
        text += f"Multiply-After Chosen Duration: {summary.get('multiply_after_best_duration', 'N/A')} ns\n"
        text += f"Delivered Gap: {summary.get('delivered_gap', 'N/A')}\n"
        text += f"Pulse Shaping Adv.: {summary.get('pulse_shaping_advantage', 'N/A')}\n"
        text += f"Duration Selection Adv.: {summary.get('duration_selection_advantage', 'N/A')}\n"
        self.h2h_output.setPlainText(text)

    def on_diag_success(self, diag_res):
        dtype, data = diag_res
        text = f"--- {dtype} Results ---\n"

        if isinstance(data, dict):
            for k, v in data.items():
                text += f"{k}: {v}\n"
        else:
             text += f"Value: {data}\n"

        self.diag_output.setPlainText(text)


    def on_error(self, error):
        print(f"Error during analysis task: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
