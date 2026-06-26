"""Optimization Panel for gradpulse GUI."""
import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt

from ui.components.mpl_widget import MatplotlibWidget
from core.worker import Worker
from gradpulse import optimize_cz, optimize_iswap, tunable_coupler_cz, ParametricCouplerProfile, CrossResonanceZXOptimizer, MultiQubitOptimizer, ParametricCZOptimizer, ActiveCancellationOptimizer, ParametricActiveCancellationOptimizer
from gradpulse.crossresonance import CrossResonanceProfile
from gradpulse.multiqubit import MultiQubitProfile
from gradpulse.viz import plot_pulse, plot_convergence

class OptimizationPanel(QWidget):
    """Optimization Panel to configure and run pulse optimizations."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()
        self.result = None
        self.loaded_profile = None

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
        self.preset_combo.addItems(["Default Parameters", "Rigetti Cepheus (Mock)", "Loaded from JSON"])
        profile_form.addRow("Preset:", self.preset_combo)

        self.load_profile_btn = QPushButton("Load Profile JSON")
        self.load_profile_btn.clicked.connect(self.load_profile_json)
        profile_form.addRow("", self.load_profile_btn)

        self.profile_status = QLabel("No profile loaded.")
        profile_form.addRow("", self.profile_status)

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
        self.gate_combo.addItems(["Parametric CZ", "iSWAP", "Tunable Coupler CZ", "Cross-Resonance ZX", "N-Qubit CZ", "Active Cancellation CZ"])
        opt_form.addRow("Target Gate:", self.gate_combo)

        self.spectral_check = QCheckBox("Spectral (CRAB/Fourier) Mode")
        opt_form.addRow("Optimization Type:", self.spectral_check)

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

        self.fidelity_combo = QComboBox()
        self.fidelity_combo.addItems(["choi", "state_transfer", "cz_data_virtualz"])
        opt_form.addRow("Fidelity Mode (N-Qubit):", self.fidelity_combo)

        opt_group.setLayout(opt_form)
        control_layout.addWidget(opt_group)

        # 3. Advanced / Robust Settings
        robust_group = QGroupBox("Robust Settings & Overrides")
        robust_form = QFormLayout()

        self.diss_scale_spin = QDoubleSpinBox()
        self.diss_scale_spin.setRange(0.0, 1.0)
        self.diss_scale_spin.setValue(1.0)
        self.diss_scale_spin.setSingleStep(0.1)
        robust_form.addRow("Dissipation Scale:", self.diss_scale_spin)

        self.robust_deph_spin = QDoubleSpinBox()
        self.robust_deph_spin.setRange(0.0, 10.0)
        self.robust_deph_spin.setValue(0.0)
        robust_form.addRow("Robust Dephasing (MHz):", self.robust_deph_spin)

        self.robust_filter_spin = QDoubleSpinBox()
        self.robust_filter_spin.setRange(0.0, 10.0)
        self.robust_filter_spin.setValue(0.0)
        robust_form.addRow("Robust Filter (MHz):", self.robust_filter_spin)

        self.n_channels_combo = QComboBox()
        self.n_channels_combo.addItems(["3", "4", "6"])
        self.n_channels_combo.setCurrentText("3")
        robust_form.addRow("Channels (Parametric):", self.n_channels_combo)

        self.coupler_phase_combo = QComboBox()
        self.coupler_phase_combo.addItems(["phase", "frequency"])
        robust_form.addRow("Coupler Phase Mode:", self.coupler_phase_combo)

        self.use_drag_check = QCheckBox("Use DRAG (CR / Parametric)")
        robust_form.addRow("", self.use_drag_check)

        self.checkpoint_spin = QSpinBox()
        self.checkpoint_spin.setRange(0, 100)
        self.checkpoint_spin.setValue(0)
        robust_form.addRow("Checkpoint Segments:", self.checkpoint_spin)

        robust_group.setLayout(robust_form)
        control_layout.addWidget(robust_group)

        # 4. Actions
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.setStyleSheet("background-color: #007acc; color: white; padding: 10px; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_optimization)
        control_layout.addWidget(self.run_btn)

        self.save_pulse_btn = QPushButton("Save Pulse JSON")
        self.save_pulse_btn.clicked.connect(self.save_pulse_json)
        self.save_pulse_btn.setEnabled(False)
        control_layout.addWidget(self.save_pulse_btn)

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

    def load_profile_json(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Calibration JSON", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    cal_dict = json.load(f)

                # Check format to determine loader
                gate = self.gate_combo.currentText()
                if "Cross-Resonance" in gate:
                     self.loaded_profile, notes = CrossResonanceProfile.from_calibration(cal_dict, (0, 1))
                elif "N-Qubit" in gate:
                     self.loaded_profile, notes = MultiQubitProfile.from_calibration(cal_dict, (0, 1, 2))
                else:
                     self.loaded_profile, notes = ParametricCouplerProfile.from_calibration(cal_dict, (0, 1))

                self.preset_combo.setCurrentText("Loaded from JSON")
                self.profile_status.setText(f"Loaded: {os.path.basename(file_name)}")
                print(f"Profile loaded successfully with notes: {notes}")
            except Exception as e:
                self.profile_status.setText(f"Error loading profile")
                print(f"Failed to load profile: {e}")

    def save_pulse_json(self):
        if not self.result:
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Pulse JSON", "pulse.json", "JSON Files (*.json)")
        if file_name:
            try:
                # Basic mock-up of what to save. In reality, it should dump the pulse, profile, and config
                gate = self.gate_combo.currentText()
                arch = "parametric_cz"
                if "Cross-Resonance" in gate:
                    arch = "cross_resonance"
                elif "N-Qubit" in gate:
                    arch = "multiqubit"

                dump_data = {
                    "architecture": arch,
                    "dt_ns": 1.0,
                    "target_gate": "cz" if "CZ" in gate else ("iswap" if "iSWAP" in gate else "zx"),
                    "waveform": self.result['best_waveform'].tolist() if hasattr(self.result['best_waveform'], "tolist") else self.result['best_waveform']
                }

                with open(file_name, 'w') as f:
                    json.dump(dump_data, f)
                print(f"Pulse saved to {file_name}")
            except Exception as e:
                print(f"Failed to save pulse: {e}")

    def run_optimization(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Optimizing...")

        # Gather inputs
        gate = self.gate_combo.currentText()
        n_slices = self.slices_spin.value()
        iterations = self.iters_spin.value()
        n_seeds = self.seeds_spin.value()
        spectral = self.spectral_check.isChecked()
        fidelity_mode = self.fidelity_combo.currentText()

        t1 = self.t1_spin.value()
        t2 = self.t2_spin.value()

        # Advanced inputs
        diss_scale = self.diss_scale_spin.value()
        r_deph = self.robust_deph_spin.value()
        r_filter = self.robust_filter_spin.value()
        n_channels = int(self.n_channels_combo.currentText())
        coupler_mode = self.coupler_phase_combo.currentText()
        use_drag = self.use_drag_check.isChecked()
        checkpoints = self.checkpoint_spin.value()

        use_preset = self.preset_combo.currentText()
        if use_preset == "Loaded from JSON" and self.loaded_profile:
            profile = self.loaded_profile
        else:
            if "Cross-Resonance" in gate:
                profile = CrossResonanceProfile(t1_ns_control=t1, t1_ns_target=t1, t2_ns_control=t2, t2_ns_target=t2)
            elif "N-Qubit" in gate:
                # N-Qubit requires specialized profile init
                profile = MultiQubitProfile(n_qubits=3, t1_ns=[t1,t1,t1], t2_ns=[t2,t2,t2], freqs_ghz=[4.8, 5.0, 4.9], anharm_mhz=[-200, -200, -200], couplings={(0,1): 10, (1,2): 10})
            else:
                profile = ParametricCouplerProfile(t1_ns_q1=t1, t1_ns_q2=t1, t2_ns_q1=t2, t2_ns_q2=t2)

        # Define the task to run in the background
        def opt_task():
            if spectral:
                if "Cross-Resonance" in gate:
                    opt = CrossResonanceZXOptimizer(profile=profile, use_drag=use_drag)
                    return opt.optimize_spectral(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds)
                elif "N-Qubit" in gate:
                    # N-Qubit does not support spectral mode, fallback to standard or raise error
                    opt = MultiQubitOptimizer(profile=profile, target_gate="cz", target_qubits=(0,1), use_drag=use_drag)
                    return opt.optimize(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, fidelity=fidelity_mode, checkpoint_segments=checkpoints)
                else:
                    target = "cz" if "CZ" in gate else ("iswap" if "iSWAP" in gate else "cz")
                    opt = ParametricCZOptimizer(profile=profile, target_gate=target, n_channels=n_channels, coupler_phase_mode=coupler_mode, use_drag=use_drag)
                    return opt.optimize_spectral(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds)
            else:
                if gate == "Parametric CZ":
                    opt = ParametricCZOptimizer(profile=profile, target_gate="cz", n_channels=n_channels, coupler_phase_mode=coupler_mode, use_drag=use_drag)
                    return opt.optimize_multi_seed(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, diss_scale=diss_scale, robust_dephasing_sigma_mhz=r_deph, robust_filter_sigma_mhz=r_filter, checkpoint_segments=checkpoints)
                elif gate == "iSWAP":
                    opt = ParametricCZOptimizer(profile=profile, target_gate="iswap", n_channels=n_channels, coupler_phase_mode=coupler_mode, use_drag=use_drag)
                    return opt.optimize_multi_seed(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, diss_scale=diss_scale, robust_dephasing_sigma_mhz=r_deph, robust_filter_sigma_mhz=r_filter, checkpoint_segments=checkpoints)
                elif gate == "Tunable Coupler CZ":
                    opt = tunable_coupler_cz(t1_ns=(t1, t1, t1), t2_ns=(t2, t2, t2), use_drag=use_drag)
                    return opt.optimize(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, checkpoint_segments=checkpoints)
                elif gate == "Cross-Resonance ZX":
                    opt = CrossResonanceZXOptimizer(profile=profile, use_drag=use_drag)
                    return opt.optimize(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, diss_scale=diss_scale)
                elif gate == "N-Qubit CZ":
                    opt = MultiQubitOptimizer(profile=profile, target_gate="cz", target_qubits=(0,1), use_drag=use_drag)
                    return opt.optimize(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, fidelity=fidelity_mode, checkpoint_segments=checkpoints)
                elif gate == "Active Cancellation CZ":
                    if isinstance(profile, ParametricCouplerProfile):
                        opt = ParametricActiveCancellationOptimizer(profile=profile, target_gate="cz", n_channels=n_channels, coupler_phase_mode=coupler_mode, use_drag=use_drag)
                        return opt.optimize_multi_seed(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, diss_scale=diss_scale, robust_dephasing_sigma_mhz=r_deph, robust_filter_sigma_mhz=r_filter, checkpoint_segments=checkpoints)
                    else:
                        opt = ActiveCancellationOptimizer(profile=profile, target_gate="cz", target_qubits=(0,1), use_drag=use_drag)
                        return opt.optimize(n_slices=n_slices, iterations=iterations, n_seeds=n_seeds, fidelity=fidelity_mode, checkpoint_segments=checkpoints)

        worker = Worker(opt_task)
        worker.signals.result.connect(self.on_optimization_success)
        worker.signals.error.connect(self.on_optimization_error)
        worker.signals.finished.connect(self.on_optimization_finished)

        self.threadpool.start(worker)

    def on_optimization_success(self, result):
        self.result = result
        print(f"Optimization finished! Best Fidelity: {result.get('best_fidelity', 'N/A')}")
        self.save_pulse_btn.setEnabled(True)

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
