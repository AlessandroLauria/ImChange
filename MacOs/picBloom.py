import sys, pathlib, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image
from PIL.ImageQt import ImageQt
import urllib.request
import numpy as np
import webbrowser
import random

sys.setrecursionlimit(5000)

# My Libraries
from Filters import Filters
from MessageBox import MessageBox
from ImportFile import ImportFile

# Global class that calls filter methods
filters = Filters()

path_to_image = "Images/"

# This function calculate the path where to find icons
# after pyinstaller export
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Current app version
version = "1.0.1"

class Application(QMainWindow, QWidget):


    def __init__(self):
        super().__init__()

        self.id = ""

        # Filters Variables
        self.lumPosition = 0
        self.contrPosition = 0
        self.satPosition = 0

        self.scalePosition = 0
        self.trasPositionX = 0
        self.trasPositionY = 0
        self.rotatePosition = 0
        self.sigma_canny = 1

        # Check operation to apply
        self.blur_filter_pressed = ""
        self.color_filter_pressed = ""
        self.edge_filter_pressed = ""
        self.transform_pressed = ""

        # Choose if apply changing to the real image
        self.apply_arith_mean = False
        self.apply_geomet_mean = False
        self.apply_harmonic_mean = False
        self.apply_median_filter = False

        self.apply_lum_filter = False
        self.apply_sat_filter = False
        self.apply_contr_filter = False

        self.apply_canny_filter = False
        self.apply_drog_filter = False

        self.apply_translation = False
        self.apply_rotation = False
        self.apply_scaling = False

        self.apply_srm = False

        # number of samples to genarate at SRM
        self.samplesSRM = 500

        self.transform_check = QCheckBox("Resize window")

        # General image variables
        self.path = ''
        self.img = ''           # preview
        self.real_img = ''      # image with real size
        self.real_img_2 = ''    # copy of real img used to apply multiple effects
        self.real_img_3 = ''
        self.test_img = ''      # image used to test filters
        self.rsize = 400        # preview dimension

        self.lbl = QLabel(self)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(resource_path(path_to_image + 'icon.png')))
        # self.showFullScreen()
        self.setStyleSheet("background-color: #423f3f; QText{color: #b4acac}")
        self.initUI()


    def initUI(self):

        self.checkUpdate()
        self.idApp()
        self.importImage(background=True)

        self.setWindowTitle('picBloom')

        self.setAcceptDrops(True)

        self.menu()
        self.show()

    # Create a file where to write an unique app_id if it not exist
    # else open the file and read the id
    def idApp(self):
        try:
            id_app = open('id_app', 'r+')
            self.id = id_app.read()

        except:
            id_app = open('id_app', 'w+')
            id_rand = random.randint(0,100000000)
            id_app.write(str(id_rand))
            self.id = id_rand

        print("id: ", self.id)
        id_app.close()

        return self.id

    # funtion that check if exist a new picbloom version
    def checkUpdate(self):
        try:
            last_version = urllib.request.urlopen("http://picbloom.altervista.org/version/version.php").read()
            last_version = np.array(last_version).astype('str')
            #last_version = last_version[1:]

            print("last version: ",last_version)
            print("current version: ", version)

            if last_version:
                if version!=last_version:
                    print("Update software")
                    alert = QMessageBox(self)
                    alert.setIcon(QMessageBox.Question)
                    alert.setStyleSheet("background-color: white; color: black; width:200;")
                    alert.setText("New Version")
                    alert.setInformativeText("There is a new picBloom version, do you want to download?")
                    alert.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
                    alert.show()
                    reply = alert.exec_()
                    if reply == QMessageBox.Yes:
                        print('download')
                        url = "http://picbloom.altervista.org/home/index.html"
                        webbrowser.open_new_tab(url)
                        # urllib.request.urlopen("http://picbloom.altervista.org/download/picBloom-macOs.zip").read()

        except:
            print("Error query")



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
    # change value. Call the function passing the specific index
    def slider(self, function, position, minimum=-127, maximum=127):
        sld = QSlider(Qt.Horizontal, self)
        sld.setMinimum(minimum)
        sld.setMaximum(maximum)
        sld.setTickInterval(position)
        sld.setFocusPolicy(Qt.NoFocus)
        # sld.setGeometry(100, 100, 100, 30)
        sld.valueChanged[int].connect(function)
        return sld

    # Make a button with text passed as parameter
    # and call a specific funtion when pressed
    def button(self,function, text):
        btn = QPushButton(text, self)
        btn.clicked.connect(function)
        return btn


    # Filters section:
    # Filters follow this work flow -
    # if "apply" variable of the filter is True, apply the filter
    # on real image that is not showed. Else use the test image
    # to show the result of the application as preview
    #
    # A variable 'Pressed' records the last filter applied
    #
    # All filters are applied using 'filters' library

    # Saturation filter
    def saturationFilter(self, index):
        if self.color_filter_pressed != "sat":
            self.img = self.test_img

        self.color_filter_pressed = "sat"

        self.satPosition = index

        if self.apply_sat_filter:
            self.real_img = filters.saturationFilter(index, self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            #self.img = self.test_img
            self.test_img = filters.saturationFilter(index, self.img)
            self.showImage(self.test_img)

    # Contrast filter
    def contrastFilter(self, index):

        if self.color_filter_pressed != "contr":
            self.img = self.test_img

        self.color_filter_pressed = "contr"

        self.contrPosition = index

        if self.apply_contr_filter:
            self.real_img = filters.contrastFilter(index, self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            #self.img = self.test_img
            self.test_img = filters.contrastFilter(index, self.img)
            self.showImage(self.test_img)


    # Luminance filter
    def luminanceFilter(self, index):

        if self.color_filter_pressed != "lum":
            self.img = self.test_img

        self.color_filter_pressed = "lum"


        self.lumPosition = index

        if self.apply_lum_filter:
            self.real_img = filters.luminanceFilter(index, self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            #self.img = self.test_img
            self.test_img = filters.luminanceFilter(index,self.img)
            self.showImage(self.test_img)

    # Arithmetic Mean Filter
    def arithmeticMeanFilter(self):

        self.blur_filter_pressed = "arith"

        if self.apply_arith_mean:
            print("apply arithmetic")
            self.real_img = filters.arithmeticMeanFilter(self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.img = filters.arithmeticMeanFilter(self.img)
            self.showImage(self.img)


    # Geometric Mean Filter
    def geometricMeanFilter(self):

        self.blur_filter_pressed = "geomet"

        if self.apply_geomet_mean:
            print("apply gometric")
            self.real_img = filters.geometricMeanFilter(self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.img = filters.geometricMeanFilter(self.img)
            self.showImage(self.img)

    def harmonicMeanFilter(self):

        self.blur_filter_pressed = "harmonic"

        if self.apply_harmonic_mean:
            print("apply harmonic")
            self.real_img = filters.harmonicMeanFilter(self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.img = filters.harmonicMeanFilter(self.img)
            self.showImage(self.img)

    def medianFilter(self):

        self.blur_filter_pressed = "median"

        if self.apply_median_filter:
            print("apply median")
            self.real_img = filters.medianFilter(self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.img = filters.medianFilter(self.img)
            self.showImage(self.img)

    def cannyFilter(self, sigma = 1):

        self.edge_filter_pressed = "canny"
        self.sigma_canny = sigma

        self.img = self.real_img
        self.resize(self.rsize)

        if self.apply_canny_filter:
            self.real_img = filters.cannyEdgeDetectorFilter(self.real_img, sigma)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.img = filters.cannyEdgeDetectorFilter(self.img, sigma)
            self.showImage(self.img)

    def drogFilter(self):

        self.edge_filter_pressed = "drog"

        if self.apply_drog_filter:
            self.real_img = filters.drogEdgeDetectorFilter(self.real_img)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
            print("Drog applied")
        else:
            self.img = filters.drogEdgeDetectorFilter(self.img)
            self.showImage(self.img)

    def translateX(self, index):
        if self.transform_pressed != "translateX":
            self.img = self.test_img
            self.real_img_2 = self.real_img_3

        self.transform_pressed = "translateX"

        self.trasPositionX = index
        self.translate(index, 0)

    def translateY(self, index):
        if self.transform_pressed != "translateY":
            self.img = self.test_img
            self.real_img_2 = self.real_img_3

        self.transform_pressed = "translateY"

        self.trasPositionY = index
        self.translate(0, index)

    def translate(self, x, y):

        self.trasPositionX = x
        self.trasPositionY = y

        if self.apply_translation:
            print("apply translation")
            self.real_img = self.real_img_3 #filters.translate(self.real_img, x, y)
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.test_img = self.img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.test_img = filters.translate(self.img, x, y)
            self.real_img_3 = filters.translate(self.real_img_2, x, y)
            self.showImage(self.test_img)

    def rotate(self, angle):

        if self.transform_pressed != "rotate":
            self.img = self.test_img
            self.real_img_2 = self.real_img_3

        self.transform_pressed = "rotate"

        self.rotatePosition = angle

        if self.apply_rotation:
            self.real_img = self.real_img_3 #filters.rotate(self.real_img, angle, resize = self.transform_check.isChecked())
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.test_img = filters.rotate(self.img, angle, resize = self.transform_check.isChecked())
            self.real_img_3 = filters.rotate(self.real_img_2, angle, resize=self.transform_check.isChecked())
            self.showImage(self.test_img)


    def scaling(self, index):

        if self.transform_pressed != "scaling":
            self.img = self.test_img
            self.real_img_2 = self.real_img_3

        self.transform_pressed = "scaling"

        self.scalePosition = index

        if self.apply_scaling:
            self.real_img = self.real_img_3 #filters.scaling(self.real_img, self.scalePosition, self.transform_check.isChecked())
            self.real_img_2 = self.real_img
            self.real_img_3 = self.real_img
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
        else:
            self.test_img = filters.scaling(self.img, index, self.transform_check.isChecked())
            self.real_img_3 = filters.scaling(self.real_img_2, index, self.transform_check.isChecked())
            self.showImage(self.test_img)

    # Used to load the Image in the label and set related parameters
    def importImage(self, drag=False, background=False):
        imp = ImportFile()

        if background:
            self.path = resource_path('Images/camera.png')
        elif not drag:
            self.path = imp.openFileNameDialog()
            if self.path == "wrong path": self.path = resource_path('Images/camera.png')
            print("path: ", self.path)


        self.real_img = Image.open(self.path)
        self.real_img_2 = self.real_img
        self.real_img_3 = self.real_img
        self.img = self.real_img
        self.resize(self.rsize)
        self.test_img = self.img
        self.showImage(self.img)


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
        srmAct = self.SRMButton()
        transAct = self.transformsButton()
        importAct = self.importButton()
        self.statusBar().setStyleSheet("background-color: #201e1e; color: #cccaca; border: 1px #201e1e;")

        self.menuBar().setStyleSheet("background-color: #201e1e; color: #cccaca; border: 1px #3a3636")
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAct)
        fileMenu.addAction(importAct)
        fileMenu.addAction(exitAct)
        fileMenu = menubar.addMenu('&Functions')
        fileMenu.addAction(colorsAct)
        fileMenu.addAction(filtersAct)
        fileMenu.addAction(edgesAct)
        fileMenu.addAction(srmAct)
        fileMenu.addAction(transAct)

        toolbar = self.addToolBar('Toolbar')
        toolbar.orientation()
        toolbar.setStyleSheet("background-color: #201e1e; color: #cccaca; border: 1px #201e1e; padding: 3px;")
        toolbar.addAction(saveAct)
        toolbar.addAction(colorsAct)
        toolbar.addAction(filtersAct)
        toolbar.addAction(edgesAct)
        toolbar.addAction(srmAct)
        toolbar.addAction(transAct)
        toolbar.addAction(exitAct)

    # Buttons showed in toolbar and menu
    def importButton(self):
        importAct = QAction(QIcon(resource_path(path_to_image + 'import.png')), 'Import image', self)
        importAct.setShortcut('ctrl+i')
        importAct.triggered.connect(self.importImage)
        return importAct

    def transformsButton(self):
        transAct = QAction(QIcon(resource_path(path_to_image + 'transform.png')), 'Transformations', self)
        transAct.setShortcut('ctrl+t')
        transAct.setStatusTip('Translate, Rotation and Scaling')
        transAct.triggered.connect(self.transformsBox)
        return transAct

    def SRMButton(self):
        srmAct = QAction(QIcon(resource_path(path_to_image + 'region.png')), 'SRM', self)
        srmAct.setStatusTip('Apply Segmentation Region Merging')
        srmAct.triggered.connect(self.SRMBox)
        return srmAct

    def edgesButton(self):
        edgesAct = QAction(QIcon(resource_path(path_to_image + 'edge.png')), 'Blur Filters', self)
        edgesAct.setShortcut('ctrl+e')
        edgesAct.setStatusTip('Apply Edge Detection Filter')
        edgesAct.triggered.connect(self.cannyFilterBox)
        return edgesAct

    def filtersButton(self):
        filterAct = QAction(QIcon(resource_path(path_to_image + 'blur.png')), 'Blur Filters', self)
        filterAct.setShortcut('ctrl+f')
        filterAct.setStatusTip('Apply Blur Filters')
        filterAct.triggered.connect(self.blurFilterBox)
        return filterAct

    def colorsButton(self):
        colorsAct = QAction(QIcon(resource_path(path_to_image + 'sliders.png')), 'Color change', self)
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
        sliderS = self.slider(self.scaling, self.scalePosition, -360, 360)
        rotText = QLabel("Rotate", self)
        sliderR = self.slider(self.rotate, self.rotatePosition)

        widget = [trasText, sliderTX, sliderTY, scalText, sliderS, rotText, sliderR, self.transform_check]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyTransforms)
        result.exec_()


    def statisticalRegionMerging(self):

        if self.apply_srm:
            self.img = filters.statisticalRegionMerging(self.img, int(self.samplesSRM))
            self.showImage(self.img)
            self.real_img = filters.statisticalRegionMerging(self.real_img, int(self.samplesSRM))
        else:
            self.test_img = filters.statisticalRegionMerging(self.img, int(self.samplesSRM))
            self.showImage(self.test_img)


    def blurFilterBox(self):
        arith_mean_btn = self.button(self.arithmeticMeanFilter, "Arithmetic mean Blur Filter")
        geomet_mean_btn = self.button(self.geometricMeanFilter, "Geometric mean Blur Filter")
        harmonic_mean_btn = self.button(self.harmonicMeanFilter, "Harmonic mean Blur Filter")
        median_filter_btn = self.button(self.medianFilter, "Median Blur Filter")
        widget = [arith_mean_btn, geomet_mean_btn, harmonic_mean_btn, median_filter_btn]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyBlurFilter)
        result.exec_()

    def cannyFilterBox(self):
        canny_text = QLabel("Sigma Canny", self)
        #sig = QLabel(str(self.sigma_canny), self)
        #canny_btn = self.button(self.cannyFilter, "Canny")
        drog_btn = self.button(self.drogFilter, "Drog")
        sigma_slider = self.slider(self.cannyFilter, self.sigma_canny, minimum=1,maximum=10)
        widget = [canny_text, sigma_slider, drog_btn]
        result = MessageBox(widget, None)
        result.setDefaultButton(QMessageBox.Apply)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyEdgeDetectionFilter)
        result.exec_()

    def SRMBox(self):
        text = QLabel("Nearness", self)

        samples = QLineEdit()
        samples.setValidator(QIntValidator())
        samples.setMaxLength(4)
        samples.setAlignment(Qt.AlignCenter)
        samples.textChanged.connect(self.setNumberSamples)
        samples.setPlaceholderText("default: "+str(self.samplesSRM))
        srm_btn = self.button(self.statisticalRegionMerging, "Preview")

        widget = [text, samples, srm_btn]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applySRM)
        result.exec_()

    def setNumberSamples(self, text):
        self.samplesSRM = text

    def colorBox(self):

        lumtext = QLabel("Luminance", self)
        sliderL = self.slider(self.luminanceFilter, self.lumPosition)
        contrtext = QLabel("Contrast", self)
        sliderC = self.slider(self.contrastFilter, self.contrPosition)
        sattext = QLabel("Saturation", self)
        sliderS = self.slider(self.saturationFilter, self.satPosition, maximum=255, minimum=-255)

        widget = [lumtext, sliderL, contrtext, sliderC, sattext, sliderS]
        result = MessageBox(widget, None)
        result.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        result.buttonClicked.connect(self.applyColors)
        result.exec_()

    def saveButton(self):
        saveAct = QAction(QIcon(resource_path(path_to_image + 'save.png')), 'Save', self)
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
            #if self.color_filter_pressed == "lum":
            self.apply_lum_filter = True
            self.luminanceFilter(self.lumPosition)
            self.apply_lum_filter = False

            #if self.color_filter_pressed == "sat":
            self.apply_sat_filter = True
            self.saturationFilter(self.satPosition)
            self.apply_sat_filter = False

            #if self.color_filter_pressed == "contr":
            self.apply_contr_filter = True
            self.contrastFilter(self.contrPosition)
            self.apply_contr_filter = False

        else:
            self.color_filter_pressed = ""
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
            print("Not modified")

    def applyTransforms(self, btn):
        self.transform_check = QCheckBox("Resize window")
        if btn.text() == "Apply":
            if self.transform_pressed == "translateX" or self.transform_pressed == "translateY":
                self.apply_translation = True
                self.translate(self.trasPositionX, self.trasPositionY)
                self.apply_translation = False

            if self.transform_pressed == "scaling":
                self.apply_scaling = True
                self.scaling(self.scalePosition)
                self.apply_scaling = False

            if self.transform_pressed == "rotate":
                self.apply_rotation = True
                self.rotate(self.rotatePosition)
                self.apply_rotation = False

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
            self.showImage(self.img)
            print("Not modified")

    def applySRM(self, btn):
        if btn.text() == "Apply":
            self.apply_srm = True
            self.statisticalRegionMerging()
            self.apply_srm = False

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
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

            if self.blur_filter_pressed == "harmonic":
                self.apply_harmonic_mean = True
                self.harmonicMeanFilter()
                self.apply_harmonic_mean = False

            if self.blur_filter_pressed == "median":
                self.apply_median_filter = True
                self.medianFilter()
                self.apply_median_filter = False

        else:
            self.img = self.real_img
            self.resize(self.rsize)
            self.test_img = self.img
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
            self.test_img = self.img
            self.showImage(self.img)
            print("Not modified")


    def saveImage(self, btn):
        path = ImportFile.saveFileDialog(self);
        if btn.text() == "Save":
            if(path == "wrong path"):
                return
            else:
                print("saved")

                tmp = path.split('/')
                tmp = tmp[-1]
                print(tmp);
                extension = tmp.split('.')

                if len(extension) < 2:
                    self.real_img.save(path+'.png', quality=100)
                    return
                if extension[len(extension)-1]!='png' and extension[len(extension)-1]!='jpg' and extension[len(extension)-1]!='jpeg':
                    return

                self.real_img.save(path, quality=100)

    def exitButton(self):
        exitAct = QAction(QIcon(resource_path(path_to_image + 'exit.png')), 'Exit', self)
        exitAct.setShortcut('ctrl+q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.close)
        return exitAct

    '''
    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):

        data = e.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            filepath = str(urls[0].path())[1:]
            print("filepath: ",filepath)
            self.path = filepath
            self.importImage(drag=True)



        if e.mimeData().hasUrls:
            e.setDropAction(Qt.MoveAction)
            e.accept()

            newText = []
            for url in e.mimeData().urls():
                newText += str(url.toLocalFile())

            self.path = '/'+ newText[0]
            self.importImage(drag=True)

        else:
            print("file ignored")
            e.ignore()

        self.showImage(self.img)
    '''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Application()
    sys.exit(app.exec_())
