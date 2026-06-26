from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QSpinBox,
    QPushButton, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt

from core.worker import Worker
from gradpulse.rb import interleaved_rb, gate_superoperator

class RBPanel(QWidget):
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

        rb_group = QGroupBox("Interleaved RB Settings")
        rb_form = QFormLayout()

        self.rb_sequences = QSpinBox()
        self.rb_sequences.setRange(10, 1000)
        self.rb_sequences.setValue(40)
        rb_form.addRow("Sequences:", self.rb_sequences)

        self.run_rb_btn = QPushButton("Run Simulated IRB")
        self.run_rb_btn.clicked.connect(self.run_rb)
        rb_form.addRow("", self.run_rb_btn)

        self.run_unitarity_btn = QPushButton("Run Unitarity RB")
        self.run_unitarity_btn.clicked.connect(self.run_unitarity)
        rb_form.addRow("", self.run_unitarity_btn)

        rb_group.setLayout(rb_form)
        control_layout.addWidget(rb_group)

        self.rb_output = QTextEdit()
        self.rb_output.setReadOnly(True)
        self.rb_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.rb_output)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

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

    def run_unitarity(self):
        self.run_unitarity_btn.setEnabled(False)
        self.run_unitarity_btn.setText("Running...")
        self.rb_output.clear()

        def unitarity_task():
            import sys
            import os
            from pathlib import Path
            # Path to examples directory
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            unitarity_script = root_dir / "examples" / "unitarity_rb.py"

            if not unitarity_script.exists():
                return "Error: Could not find examples/unitarity_rb.py"

            import subprocess
            result = subprocess.run([sys.executable, str(unitarity_script)], capture_output=True, text=True)
            return result.stdout

        worker = Worker(unitarity_task)
        worker.signals.result.connect(self.on_unitarity_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_unitarity_btn, "Run Unitarity RB"))

        self.threadpool.start(worker)

    def on_rb_success(self, rb_res):
        text = "--- IRB Results ---\n"
        text += f"Naive CZ Error: {rb_res.get('r_cz_naive', 'N/A')}\n"
        text += f"Leakage-Aware CZ Error: {rb_res.get('r_cz_leakage_aware', 'N/A')}\n"
        text += f"CZ Fidelity (IRB): {rb_res.get('f_cz_irb', 'N/A')}\n"
        text += f"Leakage/Clifford (L1): {rb_res.get('leakage_per_clifford_L1', 'N/A')}\n"
        self.rb_output.setPlainText(text)

    def on_unitarity_success(self, result_text):
        self.rb_output.setPlainText("--- Unitarity RB Script Output ---\n\n" + result_text)

    def on_error(self, error):
        print(f"RB Error: {error[1]}")
        self.rb_output.setPlainText(f"Error: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
