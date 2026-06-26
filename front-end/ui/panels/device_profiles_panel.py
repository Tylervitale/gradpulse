import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget, QScrollArea
)
from PyQt6.QtCore import Qt

from gradpulse.profiles import ParametricCouplerProfile
from gradpulse.multiqubit import MultiQubitProfile
from gradpulse.crossresonance import CrossResonanceProfile

class DeviceProfilesPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        # We start by defaulting to a ParametricCouplerProfile if one isn't loaded
        if self.main_window.active_profile is None:
            self.main_window.active_profile = ParametricCouplerProfile()
            self.main_window.active_profile_type = "Parametric Coupler"
        self.initUI()
        self.populate_fields()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Profile Type Selection
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Profile Type:"))
        self.profile_type_combo = QComboBox()
        self.profile_type_combo.addItems(["Parametric Coupler", "Cross-Resonance", "Multi-Qubit (3 Qubits)"])
        self.profile_type_combo.setCurrentText(self.main_window.active_profile_type)
        self.profile_type_combo.currentIndexChanged.connect(self.on_profile_type_changed)
        top_layout.addWidget(self.profile_type_combo)

        self.load_btn = QPushButton("Load from JSON (Calibration)")
        self.load_btn.clicked.connect(self.load_profile_json)
        top_layout.addWidget(self.load_btn)

        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Scroll Area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        self.form_layout = QFormLayout(content_widget)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.clicked.connect(self.apply_changes)
        bottom_layout.addWidget(self.apply_btn)

        self.status_label = QLabel("")
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

        self.fields = {}

    def on_profile_type_changed(self):
        new_type = self.profile_type_combo.currentText()
        if new_type == "Parametric Coupler":
            self.main_window.active_profile = ParametricCouplerProfile()
        elif new_type == "Cross-Resonance":
            self.main_window.active_profile = CrossResonanceProfile()
        elif new_type == "Multi-Qubit (3 Qubits)":
            self.main_window.active_profile = MultiQubitProfile(n_qubits=3, freqs_ghz=[4.8, 5.0, 4.9], anharm_mhz=[-200, -200, -200], t1_ns=[30000]*3, t2_ns=[25000]*3, couplings={(0,1): 12.0, (1,2): 12.0})
        self.main_window.active_profile_type = new_type
        self.populate_fields()
        self.status_label.setText(f"Switched to default {new_type} profile.")

    def populate_fields(self):
        # Clear existing layout
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.fields.clear()

        profile = self.main_window.active_profile

        # Build UI based on dataclass fields dynamically where possible, or explicitly
        if self.main_window.active_profile_type == "Parametric Coupler":
            self._add_spinbox("t1_ns_q1", "T1 Qubit 1 (ns)", profile.t1_ns_q1, 1000, 500000)
            self._add_spinbox("t1_ns_q2", "T1 Qubit 2 (ns)", profile.t1_ns_q2, 1000, 500000)
            self._add_spinbox("t2_ns_q1", "T2 Qubit 1 (ns)", profile.t2_ns_q1, 1000, 500000)
            self._add_spinbox("t2_ns_q2", "T2 Qubit 2 (ns)", profile.t2_ns_q2, 1000, 500000)
            self._add_doublespinbox("freq_ghz_q1", "Freq Qubit 1 (GHz)", profile.freq_ghz_q1, 1.0, 10.0, 4)
            self._add_doublespinbox("freq_ghz_q2", "Freq Qubit 2 (GHz)", profile.freq_ghz_q2, 1.0, 10.0, 4)
            self._add_doublespinbox("anharm_ghz_q1", "Anharm Qubit 1 (GHz)", profile.anharm_ghz_q1, -1.0, 1.0, 4)
            self._add_doublespinbox("anharm_ghz_q2", "Anharm Qubit 2 (GHz)", profile.anharm_ghz_q2, -1.0, 1.0, 4)
            self._add_doublespinbox("g_max_mhz", "g_max (MHz)", profile.g_max_mhz, 0.0, 500.0, 2)
            self._add_spinbox("n_levels", "Truncation Levels", profile.n_levels, 3, 6)

        elif self.main_window.active_profile_type == "Cross-Resonance":
            self._add_spinbox("t1_ns_control", "T1 Control (ns)", profile.t1_ns_control, 1000, 500000)
            self._add_spinbox("t1_ns_target", "T1 Target (ns)", profile.t1_ns_target, 1000, 500000)
            self._add_spinbox("t2_ns_control", "T2 Control (ns)", profile.t2_ns_control, 1000, 500000)
            self._add_spinbox("t2_ns_target", "T2 Target (ns)", profile.t2_ns_target, 1000, 500000)
            self._add_doublespinbox("freq_ghz_control", "Freq Control (GHz)", profile.freq_ghz_control, 1.0, 10.0, 4)
            self._add_doublespinbox("freq_ghz_target", "Freq Target (GHz)", profile.freq_ghz_target, 1.0, 10.0, 4)
            self._add_doublespinbox("anharm_ghz_control", "Anharm Control (GHz)", profile.anharm_ghz_control, -1.0, 1.0, 4)
            self._add_doublespinbox("anharm_ghz_target", "Anharm Target (GHz)", profile.anharm_ghz_target, -1.0, 1.0, 4)
            self._add_doublespinbox("j_coupling_mhz", "J Coupling (MHz)", profile.j_coupling_mhz, 0.0, 100.0, 2)
            self._add_spinbox("n_levels", "Truncation Levels", profile.n_levels, 3, 6)

        elif self.main_window.active_profile_type == "Multi-Qubit (3 Qubits)":
            # Just showing a few fields for multi-qubit to keep it simple, as lists are trickier
            self.form_layout.addRow(QLabel("Multi-Qubit Profiles are best loaded from JSON or defined in code."))
            self._add_spinbox("n_levels", "Truncation Levels", profile.n_levels, 3, 6)

    def _add_spinbox(self, attr_name, label_text, value, min_val, max_val):
        sb = QSpinBox()
        sb.setRange(int(min_val), int(max_val))
        sb.setValue(int(value))
        self.form_layout.addRow(label_text, sb)
        self.fields[attr_name] = sb

    def _add_doublespinbox(self, attr_name, label_text, value, min_val, max_val, decimals):
        sb = QDoubleSpinBox()
        sb.setDecimals(decimals)
        sb.setRange(min_val, max_val)
        sb.setValue(float(value))
        self.form_layout.addRow(label_text, sb)
        self.fields[attr_name] = sb

    def apply_changes(self):
        profile = self.main_window.active_profile
        for attr_name, widget in self.fields.items():
            setattr(profile, attr_name, widget.value())
        self.status_label.setText("Profile updated.")

    def load_profile_json(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Calibration JSON", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    cal_dict = json.load(f)

                ptype = self.profile_type_combo.currentText()
                if ptype == "Parametric Coupler":
                    self.main_window.active_profile, _ = ParametricCouplerProfile.from_calibration(cal_dict, (0, 1))
                elif ptype == "Cross-Resonance":
                    self.main_window.active_profile, _ = CrossResonanceProfile.from_calibration(cal_dict, (0, 1))
                elif ptype == "Multi-Qubit (3 Qubits)":
                    self.main_window.active_profile, _ = MultiQubitProfile.from_calibration(cal_dict, (0, 1, 2))

                self.populate_fields()
                self.status_label.setText(f"Loaded profile from {os.path.basename(file_name)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load profile:\n{e}")
