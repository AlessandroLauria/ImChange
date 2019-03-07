import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon


class ImportFile(QWidget):


    def __init__(self):
        super().__init__()


        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)


        self.setGeometry(self.left, self.top, self.width, self.height)


    def openFileNameDialog(self):
        options = QFileDialog.Options()


        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Images (*.jpg *.png *.jpeg);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            return fileName

        return "wrong path"

    def saveFileDialog(self):
        options = QFileDialog.Options()


        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
            return fileName

        return "path sbagliato"