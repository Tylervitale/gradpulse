from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSplitter
)
from PyQt6.QtCore import Qt

from ui.components.mpl_widget import MatplotlibWidget
from core.worker import Worker
from gradpulse.viz import plot_pulse, plot_spectrogram, plot_state_heatmap, plot_bloch_trajectory

class VisualizerPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.threadpool
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Visualization Type:"))

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Pulse Envelope",
            "Pulse Spectrogram",
            "State Heatmap",
            "Bloch Trajectory"
        ])
        top_bar.addWidget(self.viz_type_combo)

        self.plot_btn = QPushButton("Render Plot")
        self.plot_btn.clicked.connect(self.render_plot)
        top_bar.addWidget(self.plot_btn)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.plot_widget = MatplotlibWidget()
        layout.addWidget(self.plot_widget)

    def render_plot(self):
        self.plot_btn.setEnabled(False)
        self.plot_btn.setText("Rendering...")

        viz_type = self.viz_type_combo.currentText()

        def plot_task():
            opt_panel = self.main_window.opt_panel
            if not opt_panel.result or 'best_waveform' not in opt_panel.result:
                raise ValueError("No active optimization result found. Please run an optimization first.")
            result = opt_panel.result

            return viz_type, result

        worker = Worker(plot_task)
        worker.signals.result.connect(self.on_plot_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.plot_btn, "Render Plot"))
        self.threadpool.start(worker)

    def on_plot_success(self, result_tuple):
        viz_type, data = result_tuple
        self.plot_widget.clear()

        try:
            if viz_type == "Pulse Envelope":
                plot_pulse(data, ax=self.plot_widget.get_axes())
            elif viz_type == "Pulse Spectrogram":
                # plot_spectrogram expects the waveform directly, not the full result dict
                waveform = data.get('best_waveform')
                plot_spectrogram(waveform, ax=self.plot_widget.get_axes())
            elif viz_type == "State Heatmap":
                plot_state_heatmap(data, ax=self.plot_widget.get_axes())
            elif viz_type == "Bloch Trajectory":
                plot_bloch_trajectory(data, ax=self.plot_widget.get_axes())

            self.plot_widget.canvas.draw()
        except Exception as e:
            self.on_error((None, str(e), None))

    def on_error(self, error):
        print(f"Visualization Error: {error[1]}")
        # Could show a QMessageBox here

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)
