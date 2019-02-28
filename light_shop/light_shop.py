import sys, os, pathlib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image
from PIL.ImageQt import ImageQt
#from matplotlib import numpy as np

# My Libraries
from Filters import Filters
from MessageBox import MessageBox

# Global class with I call filters method
filters = Filters()

path_to_image = "Images/"


class Application(QMainWindow, QWidget):


    def __init__(self):
        super().__init__()

        # Filters Variables
        self.lumPosition = 0
        self.contrPosition = 0
        self.satPosition = 0

        self.scalePosition = 0
        self.trasPositionX = 0
        self.trasPositionY = 0

        # Check operation to apply
        self.blur_filter_pressed = ""
        self.color_filter_pressed = ""
        self.edge_filter_pressed = ""
        self.transform_pressed = ""

        # Choose if apply changing to the real image
        self.apply_arith_mean = False
        self.apply_geomet_mean = False

        self.apply_lum_filter = False
        self.apply_sat_filter = False
        self.apply_contr_filter = False

        self.apply_canny_filter = False
        self.apply_drog_filter = False

        self.apply_translation = False
        self.apply_rotation = False
        self.apply_scaling = False

        # General image variables
        self.path = '/Users/alessandrolauria/Desktop/LightShop/light_shop/Images/minions.jpg'
        self.img = ''           # preview
        self.real_img = ''      # immagine reale
        self.test_img = ''      # immagine copia di img usata per preview operazioni
        self.rsize = 400        # dimesione preview

        self.lbl = QLabel(self)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(path_to_image + 'icon.png'))
        # self.showFullScreen()
        self.setStyleSheet("background-color: #423f3f; QText{color: #b4acac}")
        self.initUI()

    def initUI(self):

        self.importImage()

        self.setWindowTitle('LightShop')

        self.setAcceptDrops(True)

        self.menu()
        self.show()


    # Used to show on the label the image passed as parameter
    def showImage(self, img):
        rgba_img = img.convert("RGBA")
        qim = ImageQt(rgba_img)
        pix = QPixmap.fromImage(qim)
        self.lbl.deleteLater()
        self.lbl = QLabel(self)
        self.lbl.setPixmap(pix)
        self.lbl.resize(pix.width(), pix.height())
        width = self.geometry().width()
        height = self.geometry().height()
        self.lbl.move(width/2 - pix.width()/2, height/2 - pix.height()/2)
        self.lbl.updateGeometry()
        self.lbl.update()
        self.update()
        self.lbl.show()

    # Check if the RGB values go outline
    def mantainInRange(self, red, green, blue):
        if red >= 255: red = 255
        if green >= 255: green = 255
        if blue >= 255: blue = 255
        if red <= 0: red = 0
        if green <= 0: green = 0
        if blue <= 0: blue = 0
        return tuple([red, green, blue])

    # Create a slider that call a specific function when the index of the slider
    # change value. Call the function with passing the specific index
    def slider(self, function, position, minimum=-127, maximum=127):
        sld = QSlider(Qt.Horizontal, self)
        sld.setMinimum(minimum)
        sld.setMaximum(maximum)
        sld.setTickInterval(position)
        sld.setFocusPolicy(Qt.NoFocus)
        # sld.setGeometry(100, 100, 100, 30)
        sld.valueChanged[int].connect(function)
        return sld

    def button(self,function, text):
        btn = QPushButton(text, self)
        btn.clicked.connect(function)
        return btn

    # Saturation filter
    def saturationFilter(self, index):
        self.color_filter_pressed = "sat"
        self.lumPosition = 0
        self.contrPosition = 0

        self.satPosition = index

        if self.apply_sat_filter:
            self.real_img = filters.saturationFilter(index, self.real_img)
            self.img = filters.saturationFilter(index, self.img)
            self.showImage(self.img)
        else:
            self.test_img = filters.saturationFilter(index, self.img)
            self.showImage(self.test_img)

    # Contrast filter
    # funzione sigmoide
    def contrastFilter(self, index):
        self.color_filter_pressed = "contr"
        self.lumPosition = 0
        self.satPosition = 0

        self.contrPosition = index

        if self.apply_contr_filter:
            self.real_img = filters.contrastFilter(index, self.real_img)
            self.img = filters.contrastFilter(index, self.img)
            self.showImage(self.img)
        else:
            self.test_img = filters.contrastFilter(index, self.img)
            self.showImage(self.test_img)


    # Luminance filter
    def luminanceFilter(self, index):
        self.color_filter_pressed = "lum"
        self.contrPosition = 0
        self.satPosition = 0

        self.lumPosition = index

        if self.apply_lum_filter:
            self.real_img = filters.luminanceFilter(index, self.real_img)
            self.img = filters.luminanceFilter(index,self.img)
            self.showImage(self.img)
        else:
            self.test_img = filters.luminanceFilter(index,self.img)
            self.showImage(self.test_img)

    # Arithmetic Mean Filter
    def arithmeticMeanFilter(self):

        self.blur_filter_pressed = "arith"

        if self.apply_arith_mean:
            self.real_img = filters.arithmeticMeanFilter(self.real_img)
            self.showImage(self.img)
        else:
            self.img = filters.arithmeticMeanFilter(self.img)
            self.showImage(self.img)


    # Geometric Mean Filter
    def geometricMeanFilter(self):

        self.blur_filter_pressed = "geomet"

        if self.apply_arith_mean:
            self.real_img = filters.geometricMeanFilter(self.real_img)
            self.showImage(self.img)
        else:
            self.img = filters.geometricMeanFilter(self.img)
            self.showImage(self.img)

    def cannyFilter(self):

        self.edge_filter_pressed = "canny"

        if self.apply_canny_filter:
            self.real_img = filters.cannyEdgeDetectorFilter(self.real_img)
            self.showImage(self.img)
        else:
            self.img = filters.cannyEdgeDetectorFilter(self.img)
            self.showImage(self.img)

    def drogFilter(self):

        self.edge_filter_pressed = "drog"

        if self.apply_drog_filter:
            self.real_img = filters.drogEdgeDetectorFilter(self.real_img)
            self.showImage(self.img)
            print("Drog applied")
        else:
            self.img = filters.drogEdgeDetectorFilter(self.img)
            self.showImage(self.img)

    def translateX(self, index):
        self.trasPositionX = index
        self.translate(index, 0)

    def translateY(self, index):
        self.trasPositionY = index
        self.translate(0, index)

    def translate(self, x, y):

        self.transform_pressed = "translate"

        if self.apply_translation:
            self.real_img = filters.translate(self.real_img, x, y)
            self.showImage(self.img)
        else:
            self.test_img = filters.translate(self.img, x, y)
            self.showImage(self.test_img)

    def scaling(self, index):

        self.transform_pressed = "scaling"

        if self.apply_scaling:
            self.real_img = filters.scaling(self.real_img, self.scalePosition)
            self.showImage(self.img)
        else:
            self.test_img = filters.scaling(self.img, index)
            self.showImage(self.test_img)

    # Used to load the Image in the label and set related parameters
    def importImage(self):
        print("path: ", self.path)
        #if os.path.exists(pathlib.Path(self.path)):
        self.real_img = Image.open(self.path)
        self.img = self.real_img
        self.test_img = self.real_img
        self.resize(self.rsize)
        self.showImage(self.img)

        #else: print("Nothing imported")

    # resize dimension of the preview image
    def resize(self, width):
        wpercent = (width / float(self.img.size[0]))
        hsize = int((float(self.img.size[1]) * float(wpercent)))
        self.img = self.img.resize((width, hsize), Image.ANTIALIAS)

    # Menu, toolbar
    def menu(self):
        exitAct = self.exitButton()
        saveAct = self.saveButton()
        colorsAct = self.colorsButton()
        filtersAct = self.filtersButton()
        edgesAct = self.edgesButton()
        srmAct = self. SRMButton()
        transAct = self.transformsButton()
        self.statusBar().setStyleSheet("background-color: #201e1e; color: #cccaca; border: 1px #201e1e;")

        self.menuBar().setStyleSheet("background-color: #201e1e; color: #cccaca; border: 1px #3a3636")
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAct)
        fileMenu.addAction(exitAct)
        fileMenu = menubar.addMenu('&Edit')
        fileMenu.addAction(colorsAct)
        fileMenu.addAction(filtersAct)
        fileMenu.addAction(edgesAct)
        fileMenu.addAction(srmAct)
        fileMenu.addAction(transAct)

        toolbar = self.addToolBar('Toolbar')
        toolbar.orientation()
        toolbar.setStyleSheet("background-color: #201e1e; color: #cccaca; border: 1px #201e1e; padding: 3px;")
        toolbar.addAction(exitAct)
        toolbar.addAction(saveAct)
        toolbar.addAction(colorsAct)
        toolbar.addAction(filtersAct)
        toolbar.addAction(edgesAct)
        toolbar.addAction(srmAct)
        toolbar.addAction(transAct)

    # Buttons showed in toolbar and menu
    def transformsButton(self):
        transAct = QAction(QIcon(path_to_image + 'transform.png'), 'Transformations', self)
        transAct.setShortcut('ctrl+t')
        transAct.setStatusTip('Translate, Rotation and Scaling')
        transAct.triggered.connect(self.transformsBox)
        return transAct

    def SRMButton(self):
        srmAct = QAction(QIcon(path_to_image + 'region.png'), 'SRM', self)
        srmAct.setStatusTip('Apply Segmentation Region Merging')
        srmAct.triggered.connect(self.statisticalRegionMerging)
        return srmAct

    def edgesButton(self):
        edgesAct = QAction(QIcon(path_to_image + 'edge.png'), 'Blur Filters', self)
        edgesAct.setShortcut('ctrl+e')
        edgesAct.setStatusTip('Apply Edge Detection Filter')
        edgesAct.triggered.connect(self.cannyFilterBox)
        return edgesAct

    def filtersButton(self):
        filterAct = QAction(QIcon(path_to_image + 'blur.png'), 'Blur Filters', self)
        filterAct.setShortcut('ctrl+f')
        filterAct.setStatusTip('Apply Blur Filters')
        filterAct.triggered.connect(self.blurFilterBox)
        return filterAct

    def colorsButton(self):
        colorsAct = QAction(QIcon(path_to_image + 'sliders.png'), 'Color change', self)
        colorsAct.setShortcut('ctrl+c')
        colorsAct.setStatusTip('Change Luminance, Contrast or Saturation')
        colorsAct.triggered.connect(self.colorBox)
        return colorsAct

    # Message Box that shows operation can be done
    def transformsBox(self):
        trasText = QLabel("Translate", self)
        sliderTX = self.slider(self.translateX, self.trasPositionX)
        sliderTY = self.slider(self.translateY, self.trasPositionY)
        scalText = QLabel("Scaling", self)
        sliderS = self.slider(self.scaling, self.scalePosition)

        widget = [trasText, sliderTX, sliderTY, scalText, sliderS]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyTransforms)
        result.exec_()


    def statisticalRegionMerging(self):
        self.img = filters.statisticalRegionMerging(self.img)
        self.showImage(self.img)

    def blurFilterBox(self):
        arith_mean_btn = self.button(self.arithmeticMeanFilter, "Arithmetic Blur Filter")
        geomet_mean_btn = self.button(self.geometricMeanFilter, "Geometric Blur Filter")
        widget = [arith_mean_btn, geomet_mean_btn]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyBlurFilter)
        result.exec_()

    def cannyFilterBox(self):
        canny_btn = self.button(self.cannyFilter, "Canny")
        drog_btn = self.button(self.drogFilter, "Drog")
        widget = [canny_btn, drog_btn]
        result = MessageBox(widget, None)
        result.setDefaultButton(QMessageBox.Apply)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyEdgeDetectionFilter)
        result.exec_()


    def colorBox(self):

        lumtext = QLabel("Luminance", self)
        sliderL = self.slider(self.luminanceFilter, self.lumPosition)
        contrtext = QLabel("Contrast", self)
        sliderC = self.slider(self.contrastFilter, self.contrPosition)
        sattext = QLabel("Saturation", self)
        sliderS = self.slider(self.saturationFilter, self.satPosition)

        widget = [lumtext, sliderL, contrtext, sliderC, sattext, sliderS]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyColors)
        result.exec_()

    def saveButton(self):
        saveAct = QAction(QIcon(path_to_image + 'save.png'), 'Save', self)
        saveAct.setShortcut('ctrl+s')
        saveAct.setStatusTip('Save Image')
        saveAct.triggered.connect(self.saveDialog)
        return saveAct

    def saveDialog(self):
        print("save dialog")
        alert = QMessageBox(self)
        alert.setStyleSheet("background-color: #423f3f; color: #cccaca;")
        alert.setText("Save")
        alert.setInformativeText("Want to save image?")
        alert.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        alert.show()
        alert.buttonClicked.connect(self.saveImage)

    def applyColors(self, btn):
        if btn.text() == "Apply":
            if self.color_filter_pressed == "lum":
                self.apply_lum_filter = True
                self.luminanceFilter(self.lumPosition)
                self.apply_lum_filter = False

            if self.color_filter_pressed == "sat":
                self.apply_sat_filter = True
                self.saturationFilter(self.satPosition)
                self.apply_sat_filter = False

            if self.color_filter_pressed == "contr":
                self.apply_contr_filter = True
                self.contrastFilter(self.contrPosition)
                self.apply_contr_filter = False

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.showImage(self.img)
            print("Not modified")

    def applyTransforms(self, btn):
        if btn.text() == "Apply":
            print("applied")

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.showImage(self.img)
            print("Not modified")

    def applyBlurFilter(self, btn):
        if btn.text() == "Apply":
            if self.blur_filter_pressed == "arith":
                self.apply_arith_mean = True
                self.arithmeticMeanFilter()
                self.apply_arith_mean = False

            if self.blur_filter_pressed == "geomet":
                self.apply_geomet_mean = True
                self.geometricMeanFilter()
                self.apply_geomet_mean = False

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.showImage(self.img)
            print("Not modified")

    def applyEdgeDetectionFilter(self, btn):
        if btn.text() == "Apply":
            if self.edge_filter_pressed == "canny":
                self.apply_canny_filter = True
                self.cannyFilter()
                self.apply_canny_filter = False

            if self.edge_filter_pressed == "drog":
                self.apply_drog_filter = True
                self.drogFilter()
                self.apply_drog_filter = False

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.showImage(self.img)
            print("Not modified")


    def saveImage(self, btn):
        if btn.text() == "Save":
            print("saved")
            self.real_img.save("test.jpeg", quality=100)

    def exitButton(self):
        exitAct = QAction(QIcon(path_to_image + 'exit.png'), 'Exit', self)
        exitAct.setShortcut('ctrl+q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.close)
        return exitAct

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):

        data = e.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            filepath = str(urls[0].path())[1:]
            print("filepath: ",filepath)
            self.path = '/'+filepath
            self.importImage()



        if e.mimeData().hasUrls:
            e.setDropAction(Qt.MoveAction)
            e.accept()

            newText = []
            for url in e.mimeData().urls():
                newText += str(url.toLocalFile())

            self.path = newText[0]
            self.importImage()

        else:
            print("file ignored")
            e.ignore()

        self.showImage(self.img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Application()
    sys.exit(app.exec_())
