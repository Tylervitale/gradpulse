from PyQt6.QtWidgets import QVBoxLayout, QWidget
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Use the PyQt6 compatible QtAgg backend
matplotlib.use('qtagg')

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MatplotlibWidget(QWidget):
    """
    A reusable widget for embedding matplotlib plots with a navigation toolbar.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def get_canvas(self):
        return self.canvas

    def get_figure(self):
        return self.canvas.fig

    def get_axes(self):
        return self.canvas.axes

    def clear(self):
        self.canvas.axes.clear()
        self.canvas.draw()
