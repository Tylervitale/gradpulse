from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit
)
from PyQt6.QtCore import Qt

from ui.components.mpl_widget import MatplotlibWidget
from core.worker import Worker
from gradpulse import optimize_cz, optimize_iswap, tunable_coupler_cz
from gradpulse.profiles import ParametricCouplerProfile
from gradpulse.viz import plot_pulse, plot_convergence

class OptimizationPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()
        self.result = None

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout(self)

        # Left Side: Controls
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. Profile Settings
        profile_group = QGroupBox("Device Profile")
        profile_form = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Default Parameters", "Rigetti Cepheus (Mock)"])
        profile_form.addRow("Preset:", self.preset_combo)

        self.t1_spin = QSpinBox()
        self.t1_spin.setRange(1000, 200000)
        self.t1_spin.setValue(30000)
        profile_form.addRow("T1 (ns):", self.t1_spin)

        self.t2_spin = QSpinBox()
        self.t2_spin.setRange(1000, 200000)
        self.t2_spin.setValue(25000)
        profile_form.addRow("T2 (ns):", self.t2_spin)

        profile_group.setLayout(profile_form)
        control_layout.addWidget(profile_group)

        # 2. Optimizer Settings
        opt_group = QGroupBox("Optimizer Settings")
        opt_form = QFormLayout()

        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["Parametric CZ", "iSWAP", "Tunable Coupler CZ"])
        opt_form.addRow("Target Gate:", self.gate_combo)

        self.slices_spin = QSpinBox()
        self.slices_spin.setRange(10, 1000)
        self.slices_spin.setValue(150)
        opt_form.addRow("Slices:", self.slices_spin)

        self.iters_spin = QSpinBox()
        self.iters_spin.setRange(10, 5000)
        self.iters_spin.setValue(200)
        opt_form.addRow("Iterations:", self.iters_spin)

        self.seeds_spin = QSpinBox()
        self.seeds_spin.setRange(1, 10)
        self.seeds_spin.setValue(2)
        opt_form.addRow("Seeds:", self.seeds_spin)

        opt_group.setLayout(opt_form)
        control_layout.addWidget(opt_group)

        # 3. Actions
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.setStyleSheet("background-color: #007acc; color: white; padding: 10px; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_optimization)
        control_layout.addWidget(self.run_btn)

        # Right Side: Visualization
        viz_layout = QVBoxLayout()

        self.pulse_plot = MatplotlibWidget()
        viz_layout.addWidget(QLabel("Pulse Envelope"))
        viz_layout.addWidget(self.pulse_plot)

        self.conv_plot = MatplotlibWidget()
        viz_layout.addWidget(QLabel("Convergence"))
        viz_layout.addWidget(self.conv_plot)

        layout.addLayout(control_layout, 1)
        layout.addLayout(viz_layout, 2)

    def run_optimization(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Optimizing...")

        # Gather inputs
        gate = self.gate_combo.currentText()
        n_slices = self.slices_spin.value()
        iterations = self.iters_spin.value()
        n_seeds = self.seeds_spin.value()

        t1 = self.t1_spin.value()
        t2 = self.t2_spin.value()

        profile = ParametricCouplerProfile(t1_ns_q1=t1, t1_ns_q2=t1, t2_ns_q1=t2, t2_ns_q2=t2)

        # Define the task to run in the background
        def opt_task():
            if gate == "Parametric CZ":
                return optimize_cz(profile=profile, n_slices=n_slices, iterations=iterations, n_seeds=n_seeds)
            elif gate == "iSWAP":
                return optimize_iswap(profile=profile, n_slices=n_slices, iterations=iterations, n_seeds=n_seeds)
            elif gate == "Tunable Coupler CZ":
                opt = tunable_coupler_cz(t1_ns=(t1, t1, t1), t2_ns=(t2, t2, t2))
                return opt.optimize(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds)

        worker = Worker(opt_task)
        worker.signals.result.connect(self.on_optimization_success)
        worker.signals.error.connect(self.on_optimization_error)
        worker.signals.finished.connect(self.on_optimization_finished)

        self.threadpool.start(worker)

    def on_optimization_success(self, result):
        self.result = result
        print(f"Optimization finished! Best Fidelity: {result.get('best_fidelity', 'N/A')}")

        # Update Plots
        self.pulse_plot.clear()
        self.conv_plot.clear()

        plot_pulse(result, ax=self.pulse_plot.get_axes())
        self.pulse_plot.get_canvas().draw()

        plot_convergence(result, ax=self.conv_plot.get_axes())
        self.conv_plot.get_canvas().draw()

    def on_optimization_error(self, error):
        print(f"Optimization Error: {error[1]}")

    def on_optimization_finished(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Optimization")
