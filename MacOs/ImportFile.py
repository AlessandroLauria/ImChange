from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog

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
        self.setStyleSheet("background-color: #423f3f; color: #b4acac;")


    def openFileNameDialog(self):
        options = QFileDialog.Options()


        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "ImChange - Import Image", "",
                                                  "Images (*.jpg *.png *.jpeg);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            return fileName

        return "wrong path"

    def saveFileDialog(self):
        options = QFileDialog.Options()


        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "Images (*.jpg *.png *.jpeg)", options=options)
        if fileName:
            print(fileName)
            return fileName

        return "wrong path"