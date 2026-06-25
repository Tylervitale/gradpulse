import sys
import io
from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor

class StreamSignaler(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

class LogConsole(QWidget):
    """
    A widget that captures stdout/stderr and displays it in a QTextEdit.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Console Output:")
        layout.addWidget(self.label)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")
        layout.addWidget(self.text_edit)

        # Capture sys.stdout
        self.stdout_signaler = StreamSignaler()
        self.stdout_signaler.text_written.connect(self.append_text)
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout_signaler

        # Capture sys.stderr
        self.stderr_signaler = StreamSignaler()
        self.stderr_signaler.text_written.connect(self.append_text)
        self.original_stderr = sys.stderr
        sys.stderr = self.stderr_signaler

    def append_text(self, text):
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.text_edit.insertPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event):
        # Restore original stdout/stderr when widget is closed
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().closeEvent(event)
