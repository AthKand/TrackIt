from PyQt6.QtWidgets import QApplication, QWizard, QWizardPage, QVBoxLayout, QLineEdit, QLabel, QFileDialog
from PyQt6.QtCore import QDir


class EnterTrackMasterPage(QWizardPage):
    def __init__(self, parent=None):
        super(EnterTrackMasterPage, self).__init__(parent)
        self.setTitle("Enter Track Master")
        self.setLayout(QVBoxLayout())
        self.label = QLabel("Enter track master:")
        self.textbox = QLineEdit()
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.textbox)
        self.registerField("enter_track_master*", self.textbox)


class EnterMouseIdPage(QWizardPage):
    def __init__(self, parent=None):
        super(EnterMouseIdPage, self).__init__(parent)
        self.setTitle("Enter Mouse ID")
        self.setLayout(QVBoxLayout())
        self.label = QLabel("Enter mouse id:")
        self.textbox = QLineEdit()
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.textbox)
        self.registerField("enter_mouse_id*", self.textbox)


class EnterDatePage(QWizardPage):
    def __init__(self, parent=None):
        super(EnterDatePage, self).__init__(parent)
        self.setTitle("Enter Date")
        self.setLayout(QVBoxLayout())
        self.label = QLabel("Enter date of experiment (DD-MM-YY):")
        self.textbox = QLineEdit()
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.textbox)
        self.registerField("enter_date*", self.textbox)

def fopen(message, ftype):
    if ftype == 'node':
        filename, _ = QFileDialog.getOpenFileName(None, message, QDir.homePath() , "CSV Files (*.csv);;All Files (*)")
    elif ftype == 'video':
        filename, _ = QFileDialog.getOpenFileNames(None, message, QDir.homePath() , "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
    elif ftype == 'dir':
        filename = QFileDialog.getExistingDirectory(None, message, QDir.homePath())

    return filename