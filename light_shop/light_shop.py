import sys, os, pathlib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image
from PIL.ImageQt import ImageQt
from matplotlib import numpy as np

# librerie mie
from Filters import Filters
from MessageBox import MessageBox

# classe globale con cui richiamare i filtri
filters = Filters()


class Application(QMainWindow, QWidget):


    def __init__(self):
        super().__init__()

        # Variabili filtri
        self.lumPosition = 0
        self.contrPosition = 0
        self.satPosition = 0

        # Check operazione da applicare
        self.blur_filter_pressed = ""
        self.color_filter_pressed = ""

        # decide se applicare le modifiche all'immagine originale
        self.apply_arith_mean = False
        self.apply_geomet_mean = False

        self.apply_lum_filter = False
        self.apply_sat_filter = False
        self.apply_contr_filter = False

        # Variabili immagine
        self.path = '/Users/alessandrolauria/Desktop/LightShop/light_shop/Images/image.jpg'
        self.img = ''           # preview
        self.real_img = ''      # immagine reale
        self.test_img = ''      # immagine copia di img usata per preview operazioni
        self.rsize = 400        # dimesione preview

        self.lbl = QLabel(self)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('Images/icon.png'))
        # self.showFullScreen()
        self.setStyleSheet("background-color: #423f3f; QText{color: #b4acac}")
        self.initUI()

    def initUI(self):

        self.importImage()

        self.setWindowTitle('FastImage')

        self.setAcceptDrops(True)

        self.menu()
        self.show()


    # Richiamato per mostrare nel label l'immagine appena modificata
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

    # Evito che i valori RGB sforino il range
    def mantainInRange(self, red, green, blue):
        if red >= 255: red = 255
        if green >= 255: green = 255
        if blue >= 255: blue = 255
        if red <= 0: red = 0
        if green <= 0: green = 0
        if blue <= 0: blue = 0
        return tuple([red, green, blue])

    # Slider per aumentare/diminuire la luminosità
    def slider(self, function, position):
        sld = QSlider(Qt.Horizontal, self)
        sld.setMinimum(-255)
        sld.setMaximum(255)
        sld.setTickInterval(position)
        sld.setFocusPolicy(Qt.NoFocus)
        # sld.setGeometry(100, 100, 100, 30)
        sld.valueChanged[int].connect(function)
        return sld

    def button(self,function, text):
        btn = QPushButton(text, self)
        btn.clicked.connect(function)
        return btn

    # Filtro saturazione
    def saturationFilter(self, index):
        self.color_filter_pressed = "sat"
        self.lumPosition = 0
        self.contrPosition = 0
        if self.apply_lum_filter == False:
            self.img = self.real_img
            self.resize(self.rsize)
            self.img.convert('HSV')
            px = self.img.load()
            width, height = self.img.size

        else:
            self.real_img.convert('HSV')
            px = self.real_img.load()
            width, height = self.real_img.size

        print("p[100,100]: ", px[100, 100])
        result = np.zeros((height, width, 3), dtype=np.uint8)
        div = 1

        for i in range(0, height):
            for j in range(0, width):
                hue, sat, value = px[j, i]
                sat += index / div
                if sat >= 255: sat = 255
                if sat <= 0: sat = 0
                rgb = tuple([hue,sat, value])
                result[i, j] = rgb

        print("result[100,100]: ", result[100, 100])
        self.satPosition = index
        self.img = Image.fromarray(result, 'HSV')

        if self.apply_sat_filter == False:
            self.img = Image.fromarray(result, 'HSV')
            self.img.convert('RGB')
        else:
            self.real_img = Image.fromarray(result, 'HSV')
            self.real_img.convert('RGB')
            self.img = self.real_img
            self.resize(self.rsize)
        self.showImage(self.test_img)

    # Filtro contrasto
    # funzione sigmoide
    def contrastFilter(self, index):
        self.color_filter_pressed = "contr"
        self.lumPosition = 0
        self.satPosition = 0

        if self.apply_contr_filter:
            self.real_img = filters.contrastFilter(index, self.real_img)
            self.img = filters.contrastFilter(index, self.img)
            self.showImage(self.img)
        else:
            self.test_img = filters.luminanceFilter(index, self.img)
            self.showImage(self.test_img)
        '''
        if self.apply_contr_filter == False:
            self.img = self.real_img
            self.resize(self.rsize)
            px = self.img.load()
            width, height = self.img.size

        else:
            px = self.real_img.load()
            width, height = self.real_img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        factor = (259.0 * (index + 255.0)) / (255.0 * (259.0 - index))

        for i in range(0, height):
            for j in range(0, width):
                red, green, blue = px[j,i]
                red = factor * (red - 128.0) + 128.0
                green = factor * (green - 128.0) + 128.0
                blue = factor * (blue - 128.0) + 128.0
                rgb = self.mantainInRange(red, green, blue)
                result[i, j] = rgb

        self.contrPosition = index
        if self.apply_contr_filter == False:
            self.img = Image.fromarray(result, 'RGB')
        else:
            self.real_img = Image.fromarray(result, 'RGB')
            self.img = self.real_img
            self.resize(self.rsize)
        self.showImage(self.test_img)
        '''

    # filtro luminosità
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

    # filtro di media aritmetica
    def arithmeticMeanFilter(self):

        self.blur_filter_pressed = "arith"
        if self.apply_arith_mean == False:
            self.img = self.real_img
            self.resize(self.rsize)
            px = self.img.load()
            width, height = self.img.size
        else:
            px = self.real_img.load()
            width, height = self.real_img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(3, height - 3):
            for j in range(3, width - 3):
                sum_red, sum_green, sum_blue = (0, 0, 0)
                for n in range(i - 3, i + 3):
                    for m in range(j - 3, j + 3):
                        r, g, b = px[m, n]
                        sum_red += r
                        sum_green += g
                        sum_blue += b

                result[i, j] = tuple([sum_red / 49, sum_green / 49, sum_blue / 49])

        if self.apply_arith_mean == False:
            print("arith filter to preview")
            self.img = Image.fromarray(result, 'RGB')
        else:
            print("arith filter to real image")
            self.real_img = Image.fromarray(result, 'RGB')
            self.img = self.real_img
            self.resize(self.rsize)
        self.showImage(self.test_img)

    # filtro di media geometrica
    def geometricMeanFilter(self):

        self.blur_filter_pressed = "geomet"
        if self.apply_geomet_mean == False:
            self.img = self.real_img
            self.resize(self.rsize)
            px = self.img.load()
            width, height = self.img.size
        else:
            px = self.real_img.load()
            width, height = self.real_img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        # kernel 9x9
        for i in range(4, height - 4):
            for j in range(4, width - 4):
                prod_red, prod_green, prod_blue = (1, 1, 1)
                for n in range(i - 4, i + 4):
                    for m in range(j - 4, j + 4):
                        r, g, b = px[m, n]
                        prod_red *= r
                        prod_green *= g
                        prod_blue *= b

                result[i, j] = tuple([prod_red ** (1 / 64), prod_green ** (1 / 64), prod_blue ** (1 / 64)])

        if self.apply_geomet_mean == False:
            print("geomet filter to preview")
            self.img = Image.fromarray(result, 'RGB')
        else:
            print("geomet filter to real image")
            self.real_img = Image.fromarray(result, 'RGB')
            self.img = self.real_img
            self.resize(self.rsize)
        self.showImage(self.test_img)

    def greyScale(self):
        print("greyscale")
        # self.img = self.img.resize([100,100],0)
        self.img = self.img.convert('LA')
        self.real_img = self.img.convert('LA')
        self.showImage(self.img)

    def importImage(self):
        print("path: ", self.path)
        #if os.path.exists(pathlib.Path(self.path)):
        self.real_img = Image.open(self.path)
        self.img = self.real_img
        self.test_img = self.real_img
        self.resize(self.rsize)
        self.showImage(self.img)

        #else: print("Nothing imported")

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
        self.statusBar().setStyleSheet("background-color: #201e1e; border: 1px #201e1e; QText{border-radius: 15}")

        self.menuBar().setStyleSheet("background-color: #201e1e; border: 1px #3a3636;")
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAct)
        fileMenu.addAction(exitAct)
        fileMenu = menubar.addMenu('&Edit')
        fileMenu.addAction(colorsAct)
        fileMenu.addAction(filtersAct)

        toolbar = self.addToolBar('Toolbar')
        toolbar.orientation()
        toolbar.setStyleSheet("background-color: #201e1e; border: 1px #201e1e; padding: 3px;")
        toolbar.addAction(exitAct)
        toolbar.addAction(saveAct)
        toolbar.addAction(colorsAct)
        toolbar.addAction(filtersAct)

    def filtersButton(self):
        filterAct = QAction(QIcon('Images/filter.png'), 'Blur Filters', self)
        filterAct.setShortcut('ctrl+f')
        filterAct.setStatusTip('Apply Blur Filters')
        filterAct.triggered.connect(self.blurFilterBox)
        return filterAct

    def colorsButton(self):
        colorsAct = QAction(QIcon('Images/sliders.png'), 'Color change', self)
        colorsAct.setShortcut('ctrl+c')
        colorsAct.setStatusTip('Change Luminance, Contrast or Saturation')
        colorsAct.triggered.connect(self.colorBox)
        return colorsAct

    def blurFilterBox(self):
        arith_mean_btn = self.button(self.arithmeticMeanFilter, "Arithmetic Blur Filter")
        geomet_mean_btn = self.button(self.geometricMeanFilter, "Geometric Blur Filter")
        widget = [arith_mean_btn, geomet_mean_btn]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyBlurFilter)
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
        saveAct = QAction(QIcon('Images/save.png'), 'Save', self)
        saveAct.setShortcut('ctrl+s')
        saveAct.setStatusTip('Save Image')
        saveAct.triggered.connect(self.saveDialog)
        return saveAct

    def saveDialog(self):
        print("save dialog")
        alert = QMessageBox(self)
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

    def saveImage(self, btn):
        if btn.text() == "Save":
            print("saved")
            self.real_img.save("test.jpeg", quality=80)

    def exitButton(self):
        exitAct = QAction(QIcon('Images/exit.png'), 'Exit', self)
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
