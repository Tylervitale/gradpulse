import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt
import numpy as np

class ProjectPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("Project Management")
        form = QFormLayout()

        self.save_btn = QPushButton("Save Project State")
        self.save_btn.clicked.connect(self.save_project)
        form.addRow("Save Current Session:", self.save_btn)

        self.load_btn = QPushButton("Load Project State")
        self.load_btn.clicked.connect(self.load_project)
        form.addRow("Load Previous Session:", self.load_btn)

        self.status_label = QLabel("No project loaded.")
        form.addRow("Status:", self.status_label)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def save_project(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Project", "gradpulse_project.json", "JSON Files (*.json)")
        if not file_name:
            return

        try:
            # Gather state from MainWindow and active panels
            state = {
                "active_profile_type": self.main_window.active_profile_type,
                "profile_dict": self.main_window.active_profile.__dict__ if self.main_window.active_profile else None,
                "optimization_result": None
            }

            # If there's an active optimization result, try to save the waveform and raw param
            opt_panel = self.main_window.opt_panel
            if opt_panel.result:
                # Need to convert numpy/torch arrays to lists for JSON serialization
                result = {}
                if 'best_waveform' in opt_panel.result:
                    wf = opt_panel.result['best_waveform']
                    result['best_waveform'] = wf.tolist() if hasattr(wf, 'tolist') else wf
                if 'best_raw_param' in opt_panel.result:
                    rp = opt_panel.result['best_raw_param']
                    result['best_raw_param'] = rp.tolist() if hasattr(rp, 'tolist') else rp
                if 'best_fidelity' in opt_panel.result:
                    result['best_fidelity'] = float(opt_panel.result['best_fidelity'])
                state['optimization_result'] = result

            with open(file_name, 'w') as f:
                json.dump(state, f, indent=4)

            self.status_label.setText(f"Project saved to {os.path.basename(file_name)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")

    def load_project(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if not file_name:
            return

        try:
            with open(file_name, 'r') as f:
                state = json.load(f)

            # Restore profile
            if state.get("active_profile_type") and state.get("profile_dict"):
                ptype = state["active_profile_type"]
                pdict = state["profile_dict"]
                self.main_window.active_profile_type = ptype

                if ptype == "Parametric Coupler":
                    from gradpulse.profiles import ParametricCouplerProfile
                    self.main_window.active_profile = ParametricCouplerProfile(**pdict)
                elif ptype == "Cross-Resonance":
                    from gradpulse.crossresonance import CrossResonanceProfile
                    self.main_window.active_profile = CrossResonanceProfile(**pdict)
                elif ptype == "Multi-Qubit (3 Qubits)":
                    from gradpulse.multiqubit import MultiQubitProfile
                    # Might need special handling for dicts/lists in kwargs
                    self.main_window.active_profile = MultiQubitProfile(**pdict)

                # Update DeviceProfilesPanel UI
                self.main_window.device_profiles_panel.profile_type_combo.setCurrentText(ptype)
                self.main_window.device_profiles_panel.populate_fields()

            # Restore optimization result (partial, since we can't easily restore the full optimizer object)
            if state.get("optimization_result"):
                res = state["optimization_result"]
                # Convert lists back to numpy arrays
                if 'best_waveform' in res:
                    res['best_waveform'] = np.array(res['best_waveform'])
                if 'best_raw_param' in res:
                    res['best_raw_param'] = np.array(res['best_raw_param'])

                self.main_window.opt_panel.result = res

                # Re-plot if possible
                try:
                    from gradpulse.viz import plot_pulse
                    self.main_window.opt_panel.pulse_plot.clear()
                    plot_pulse(res, ax=self.main_window.opt_panel.pulse_plot.get_axes())
                    self.main_window.opt_panel.pulse_plot.get_canvas().draw()
                except Exception as plot_e:
                    print(f"Could not re-plot loaded pulse: {plot_e}")

            self.status_label.setText(f"Project loaded from {os.path.basename(file_name)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{e}")
