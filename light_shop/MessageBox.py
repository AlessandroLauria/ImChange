from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class MessageBox(QMessageBox):
    def __init__(self, s, *args, **kwargs):
        QMessageBox.__init__(self, *args, **kwargs)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.content = QWidget()
        scroll.setWidget(self.content)
        lay = QVBoxLayout(self.content)

        for item in s:
            lay.addWidget(item)
        self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())
        self.setStyleSheet("QScrollArea{min-width:200 px; min-height: 200px; border: 0; background-color: #201e1e;}")
