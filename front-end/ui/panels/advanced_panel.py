from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QPushButton,
    QFormLayout, QGroupBox, QSpinBox, QHBoxLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

from core.worker import Worker
from gradpulse import rl
from gradpulse.compression import compress_rle, verify_compression

class AdvancedPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        # 1. RL Agent Tab
        self.rl_tab = QWidget()
        self.init_rl_tab()
        self.tabs.addTab(self.rl_tab, "Reinforcement Learning")

        # 2. Compression Tab
        self.compress_tab = QWidget()
        self.init_compress_tab()
        self.tabs.addTab(self.compress_tab, "Waveform Compression")

        # 3. Error Mitigation (ZNE) Tab
        self.zne_tab = QWidget()
        self.init_zne_tab()
        self.tabs.addTab(self.zne_tab, "Error Mitigation (ZNE)")

        layout.addWidget(self.tabs)

    def init_rl_tab(self):
        layout = QVBoxLayout(self.rl_tab)

        group = QGroupBox("Train PPO Agent (Stable-Baselines3)")
        form = QFormLayout()

        self.rl_steps = QSpinBox()
        self.rl_steps.setRange(100, 100000)
        self.rl_steps.setValue(2048)
        form.addRow("Total Timesteps:", self.rl_steps)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.run_rl_training)
        form.addRow("", self.train_btn)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def init_compress_tab(self):
        layout = QVBoxLayout(self.compress_tab)

        group = QGroupBox("Run-Length Encoding (RLE)")
        vbox = QVBoxLayout()

        self.compress_btn = QPushButton("Compress Current Optimized Pulse")
        self.compress_btn.clicked.connect(self.run_compression)
        vbox.addWidget(self.compress_btn)

        self.compress_results = QLabel("Ready.")
        vbox.addWidget(self.compress_results)

        group.setLayout(vbox)
        layout.addWidget(group)
        layout.addStretch()

    def init_zne_tab(self):
        layout = QVBoxLayout(self.zne_tab)

        group = QGroupBox("Zero-Noise Extrapolation Configuration")
        form = QFormLayout()

        self.zne_scale = QDoubleSpinBox()
        self.zne_scale.setRange(1.0, 5.0)
        self.zne_scale.setValue(1.5)
        self.zne_scale.setSingleStep(0.1)
        form.addRow("Noise Scale Factor:", self.zne_scale)

        self.zne_btn = QPushButton("Run ZNE Pipeline")
        self.zne_btn.clicked.connect(self.run_zne)
        form.addRow("", self.zne_btn)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def run_rl_training(self):
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training...")

        steps = self.rl_steps.value()

        def train_task():
            # Mock or call actual gradpulse.rl implementation depending on import
            print(f"Starting RL training for {steps} steps on CrossResonanceEnv...")
            return rl.train_ppo(total_timesteps=steps)

        worker = Worker(train_task)
        worker.signals.result.connect(self.on_rl_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.train_btn, "Start Training"))

        self.threadpool.start(worker)

    def run_compression(self):
        opt_panel = self.main_window.opt_panel
        if not opt_panel.result or 'best_waveform' not in opt_panel.result:
            self.compress_results.setText("Error: No pulse found. Run optimization first.")
            return

        pulse = opt_panel.result['best_waveform']

        def compress_task():
            import numpy as np
            # Convert torch tensor if needed to numpy
            arr = pulse.numpy() if hasattr(pulse, "numpy") else np.array(pulse)
            compressed = compress_rle(arr)
            valid = verify_compression(arr, compressed)
            return compressed, valid, arr.nbytes

        worker = Worker(compress_task)
        worker.signals.result.connect(self.on_compress_success)
        worker.signals.error.connect(self.on_error)

        self.threadpool.start(worker)

    def run_zne(self):
        self.zne_btn.setEnabled(False)
        print(f"Starting ZNE scaling with factor {self.zne_scale.value()}...")
        scale = self.zne_scale.value()

        opt_panel = self.main_window.opt_panel
        if not opt_panel.result or 'best_waveform' not in opt_panel.result:
            print("Error: No pulse found. Run optimization first.")
            self._reset_btn(self.zne_btn, "Run ZNE Pipeline")
            return

        pulse = opt_panel.result['best_waveform']

        def zne_task():
            from gradpulse.mitigation import stretch_pulse
            import numpy as np
            arr = pulse.numpy() if hasattr(pulse, "numpy") else np.array(pulse)
            scaled = stretch_pulse(arr, scale)
            return scaled

        worker = Worker(zne_task)
        worker.signals.result.connect(self.on_zne_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.zne_btn, "Run ZNE Pipeline"))

        self.threadpool.start(worker)

    def on_zne_success(self, scaled):
        print(f"ZNE scaled pulse size: {scaled.shape}")

    def on_rl_success(self, model):
        print("RL Training Completed Successfully.")

    def on_compress_success(self, result):
        compressed, is_valid, orig_size = result
        new_size = sum(v.nbytes if hasattr(v, "nbytes") else 0 for v in compressed.values()) if isinstance(compressed, dict) else len(compressed)*8

        self.compress_results.setText(
            f"Compression Complete:\n"
            f"Original Size: {orig_size} bytes\n"
            f"Compressed Size: ~{new_size} bytes\n"
            f"Lossless Verification: {'PASS' if is_valid else 'FAIL'}"
        )
        print("Compression algorithm ran.")

    def on_error(self, error):
        print(f"Error in Advanced Features: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
