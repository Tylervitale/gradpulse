import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QStackedWidget, QLabel, QSplitter
)
from PyQt6.QtCore import Qt, QThreadPool

from ui.components.log_console import LogConsole
# Import future panels here
from ui.panels.optimization_panel import OptimizationPanel
from ui.panels.analysis_panel import AnalysisPanel
from ui.panels.hardware_panel import HardwarePanel
from ui.panels.advanced_panel import AdvancedPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GradPulse GUI")
        self.resize(1200, 800)

        # Thread pool for running background tasks
        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

        self.initUI()

    def initUI(self):
        # Main Layout: Sidebar on the left, Content on the right
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Sidebar Navigation
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: #2e2e2e;
                color: #ffffff;
                font-size: 14px;
                padding: 10px;
                border: none;
            }
            QListWidget::item {
                padding: 10px;
                border-radius: 5px;
            }
            QListWidget::item:selected {
                background-color: #007acc;
            }
        """)

        # Add Navigation Items
        nav_items = [
            "Optimization",
            "Analysis & Validation",
            "Hardware & Export",
            "Advanced Features"
        ]
        self.sidebar.addItems(nav_items)
        self.sidebar.currentRowChanged.connect(self.display_panel)

        # 2. Main Content Area (Stacked Widget + Log Console)
        content_splitter = QSplitter(Qt.Orientation.Vertical)

        # Stacked Widget for Panels
        self.stacked_widget = QStackedWidget()

        # --- Initialize Panels (Placeholder widgets for now) ---
        self.opt_panel = OptimizationPanel(self)
        self.analysis_panel = AnalysisPanel(self)
        self.hardware_panel = HardwarePanel(self)
        self.advanced_panel = AdvancedPanel(self)

        # Add to stack
        self.stacked_widget.addWidget(self.opt_panel)
        self.stacked_widget.addWidget(self.analysis_panel)
        self.stacked_widget.addWidget(self.hardware_panel)
        self.stacked_widget.addWidget(self.advanced_panel)

        # Bottom Console
        self.log_console = LogConsole()

        content_splitter.addWidget(self.stacked_widget)
        content_splitter.addWidget(self.log_console)

        # Set sizes for splitter (e.g., 70% panels, 30% console)
        content_splitter.setSizes([700, 300])

        # Add to main layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(content_splitter)

        # Set initial selection
        self.sidebar.setCurrentRow(0)

    def display_panel(self, index):
        """Switch the stacked widget view based on sidebar selection."""
        self.stacked_widget.setCurrentIndex(index)

    def get_threadpool(self):
        return self.threadpool
