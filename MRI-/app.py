from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox
from mriui import Ui_MainWindow
from phantom import phantom
import numpy as np
import qimage2ndarray
import sys
import math
import threading
from rotation import rotateX, gradientXY
from RD import recovery, decay
import pyqtgraph as pg
from PyQt5.QtWidgets import QFileDialog
from math import sin, cos, pi
import csv

MAX_CONTRAST = 2
MIN_CONTRAST = 0.1
MAX_BRIGHTNESS = 100
MIN_BRIGHTNESS = -100
SAFETY_MARGIN = 10
MAX_PIXELS_CLICKED = 3


class ApplicationWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Actions
        self.ui.comboSheppSize.currentTextChanged.connect(self.showPhantom)
        self.ui.comboViewMode.currentTextChanged.connect(self.changePhantomMode)
        self.ui.startSeq.clicked.connect(self.runSequence)
        self.ui.FlipAngle.textChanged.connect(self.setFA)
        self.ui.TimeEcho.textChanged.connect(self.setTE)
        self.ui.TimeRepeat.textChanged.connect(self.setTR)
        self.ui.btnBrowse.clicked.connect(self.browse)

        # Mouse Events
        self.ui.phantomlbl.setMouseTracking(False)
        self.ui.phantomlbl.mouseMoveEvent = self.editContrastAndBrightness
        self.ui.phantomlbl.mouseDoubleClickEvent = self.pixelClicked
        self.ui.phantomlbl.wheelEvent = self.zoomInOut

        # Scaling

        self.ui.phantomlbl.setScaledContents(True)
        self.ui.kspaceLbl.setScaledContents(True)

        # Plots
        self.ui.graphicsPlotT1.setMouseEnabled(False, False)
        self.ui.graphicsPlotT2.setMouseEnabled(False, False)

        # initialization
        self.qimg = None
        self.img = None
        self.originalPhantom = None
        self.PD = None
        self.T1 = None
        self.T2 = None
        self.inversion_time = 4.3
        self.T2_prep_time = 2
        self.TAG_frequency = 7
        self.cycles_number = 10
        self.zoomLevel = 0
        self.phantomSize = 512

        self.FA = 90
        self.cosFA = 0
        self.sinFA = 1
        self.TE = 0.001
        self.TR = 0.5
        self.x = 0
        self.y = 0

        self.pixelsClicked = [(0, 0), (0, 0), (0, 0)]
        self.pixelSelector = 0

        # For Mouse moving, changing Brightness and Contrast
        self.lastY = None
        self.lastX = None

        # For Contrast Control
        self.contrast = 1.0
        self.brightness = 0

    def browse(self):
        # Open Browse Window & Check
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", (QtCore.QDir.homePath()), "CSV (*.csv)")
        if fileName:
            # Check extension
            try:

                mat = np.genfromtxt(fileName, delimiter=',')
                row = math.floor(np.shape(mat)[0] / 3)
                col = np.shape(mat)[1]
                self.img = np.zeros([row, col])
                self.T1 = np.zeros([row, col])
                self.T2 = np.zeros([row, col])
                self.img = mat[0:row]
                self.PD = self.img
                self.originalPhantom = self.img
                self.phantomSize = row
                print(np.shape(self.img))
                self.T1 = mat[row:2 * row]
                print(np.shape(self.T1))
                self.T2 = mat[2 * row:3 * row + 1]
                print(np.shape(self.T2))
                self.showPhantomImage()
            except (IOError, SyntaxError):
                self.error('Check File Extension')

    def showPhantom(self, value):
        size = int(value)
        self.phantomSize = size
        self.ui.phantomlbl.phantomSize = size
        img = phantom(size)
        self.PD = img
        img = img * 255
        self.img = img
        self.T1 = phantom(size, 'T1')
        self.T2 = phantom(size, 'T2')
        self.originalPhantom = img
        self.pixelsClicked = [(0, 0), [0, 0], [0, 0]]
        self.showPhantomImage(self.img)

    def zoomInOut(self, event):
        direction = event.angleDelta().y() > 0
        if direction:
            self.zoomLevel = self.zoomLevel + 1
        else:
            self.zoomLevel = self.zoomLevel - 1
        # constraints
        if self.zoomLevel < 0:
            self.zoomLevel = 0
        elif self.zoomLevel > self.phantomSize -5:
            self.zoomLevel = self.phantomSize-5

        img = self.img[0 + self.zoomLevel:-self.zoomLevel, 0 + self.zoomLevel:-self.zoomLevel]
        self.showPhantomImage(img)

    def showPhantomImage(self, img):
        self.qimg = qimage2ndarray.array2qimage(img)
        # self.ui.phantomlbl.setAlignment(QtCore.Qt.AlignCenter)
        # self.ui.phantomlbl.setFixedWidth(self.phantomSize)
        # self.ui.phantomlbl.setFixedHeight(self.phantomSize)
        self.ui.phantomlbl.setPixmap(QPixmap(self.qimg))

    def changePhantomMode(self, value):

        if value == "PD":
            self.img = self.PD
        if value == "T1":
            self.img = self.T1
        if value == "T2":
            self.img = self.T2

        self.img = self.img * (255 / np.max(self.img))
        self.originalPhantom = self.img
        self.showPhantomImage()

    def editContrastAndBrightness(self, event):
        if self.lastX is None:
            self.lastX = event.pos().x()
        if self.lastY is None:
            self.lastY = event.pos().y()
            return

        currentPositionX = event.pos().x()
        if currentPositionX - SAFETY_MARGIN > self.lastX:
            self.contrast += 0.01
        elif currentPositionX < self.lastX - SAFETY_MARGIN:
            self.contrast -= 0.01

        currentPositionY = event.pos().y()
        if currentPositionY - SAFETY_MARGIN > self.lastY:
            self.brightness += 1
        elif currentPositionY < self.lastY - SAFETY_MARGIN:
            self.brightness -= 1
        # Sanity Check
        if self.contrast > MAX_CONTRAST:
            self.contrast = MAX_CONTRAST
        elif self.contrast < MIN_CONTRAST:
            self.contrast = MIN_CONTRAST
        if self.brightness > MAX_BRIGHTNESS:
            self.brightness = MAX_BRIGHTNESS
        elif self.brightness < MIN_BRIGHTNESS:
            self.brightness = MIN_BRIGHTNESS

        self.img = 128 + self.contrast * (self.originalPhantom - 128)
        self.img = np.clip(self.img, 0, 255)

        self.img = self.img + self.brightness
        self.img = np.clip(self.img, 0, 255)
        self.showPhantomImage(self.img)

        self.lastY = currentPositionY
        self.lastX = currentPositionX

    def pixelClicked(self, event):
        if self.img is None:
            self.error('Choose Phantom First')
        else:
            self.pixelSelector = self.pixelSelector + 1
            self.pixelSelector = self.pixelSelector % 3
            t1Matrix = self.T1
            t2Matrix = self.T2
            self.x = event.pos().x()
            self.y = event.pos().y()
            self.ui.phantomlbl.pixelSelector += 1
            self.ui.phantomlbl.pixelSelector = self.ui.phantomlbl.pixelSelector % 3

            xt = self.ui.phantomlbl.frameGeometry().width()
            yt = self.ui.phantomlbl.frameGeometry().height()
            x = event.pos().x() * (self.phantomSize / xt)
            y = event.pos().y() * (self.phantomSize / yt)
            x = math.floor(x)
            y = math.floor(y)
            self.pixelsClicked.append((x, y))
            self.ui.phantomlbl.pixelsClicked.append((x, y))
            if len(self.pixelsClicked) > MAX_PIXELS_CLICKED:
                self.pixelsClicked.pop(0)
            if len(self.ui.phantomlbl.pixelsClicked) > MAX_PIXELS_CLICKED:
                self.ui.phantomlbl.pixelsClicked.pop(0)
            self.update()
            # self.paintEvent(event)
            t1graph = self.ui.graphicsPlotT1
            t2gragh = self.ui.graphicsPlotT2
            t1graph.clear()
            t2gragh.clear()

            for pixelSet in self.pixelsClicked:
                x = pixelSet[0]
                y = pixelSet[1]
                if self.pixelSelector == 0:
                    color = 'r'
                if self.pixelSelector == 1:
                    color = 'b'
                if self.pixelSelector == 2:
                    color = 'y'
                t1 = t1Matrix[y][x]
                t2 = t2Matrix[y][x]
                self.plotting(color, t1 * 1000, t2 * 1000)
                self.pixelSelector += 1
                self.pixelSelector = self.pixelSelector % 3

    # def paintEvent(self, event):
    #     # create painter instance with pixmap
    #     # canvas = QPixmap(self.ui.phantomlbl)
    #     paint = QtGui.QPainter()
    #     paint.begin(self.ui.phantomlbl)
    #
    #     # set rectangle color and thickness
    #
    #     for pixelSet in self.pixelsClicked:
    #         self.x = pixelSet[0]
    #         self.y = pixelSet[1]
    #         xt = self.ui.phantomlbl.frameGeometry().width()
    #         yt = self.ui.phantomlbl.frameGeometry().height()
    #         self.x = self.x / (self.phantomSize / xt)
    #         self.y = self.y / (self.phantomSize / yt)
    #         self.x = math.floor(self.x)
    #         self.y = math.floor(self.y)
    #         if self.pixelSelector == 0:
    #             pen = QtGui.QPen(QtCore.Qt.red)
    #         if self.pixelSelector == 1:
    #             pen = QtGui.QPen(QtCore.Qt.blue)
    #         if self.pixelSelector == 2:
    #             pen = QtGui.QPen(QtCore.Qt.yellow)
    #         pen.setWidth(1)
    #         paint.setPen(pen)
    #         # draw rectangle on painter
    #         paint.drawRect(self.x - 1, self.y - 1, 50, 20)
    #         # set pixmap onto the label widget
    #         # self.ui.phantomlbl.setPixmap(canvas)
    #         self.pixelSelector += 1
    #         self.pixelSelector = self.pixelSelector % 3
    #     paint.end()

    def plotting(self, color, T1=1000, T2=45):
        t1graph = self.ui.graphicsPlotT1
        t2gragh = self.ui.graphicsPlotT2
        # theta = self.FA * pi / 180
        t = np.linspace(0, 10000, 1000)
        t1graph.plot(np.exp(-t / T1) * self.cosFA + 1 - np.exp(-t / T1), pen=pg.mkPen(color))
        t2gragh.plot(self.sinFA * np.exp(-t / T2), pen=pg.mkPen(color))

    def setFA(self, value):
        print(value)
        try:
            value = int(value)
            self.FA = value
            self.cosFA = cos(self.FA * pi / 180)
            self.sinFA = sin(self.FA * pi / 180)
        except:
            self.error("FA must be a number")

    def setTE(self, value):
        print(value)
        try:
            value = float(value)
            self.TE = value
        except:
            self.error("TE must be a float")

    def setTR(self, value):
        print(value)
        try:
            value = float(value)
            self.TR = value
        except:
            self.error("TR must be a float")

    def runSequence(self):
        if self.img is None:
            self.error('Choose a phantom first')
        else:
            threading.Thread(target=self.GRE_reconstruct_image).start()
            return

    def GRE_reconstruct_image(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        # To be Removed
        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                vectors[i, j] = vectors[i, j].dot(self.PD[i, j])

        print(vectors[30, 10])

        vectors = self.startup_cycles(vectors)
        vectors = self.TAG_prep(vectors)

        for i in range(0, round(self.phantomSize)):
            rotatedMatrix = rotateX(vectors, self.FA)
            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, self.TE)

            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            decayedRotatedMatrix[:, :, 0] = 0
            decayedRotatedMatrix[:, :, 1] = 0
            # vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
            # vectors[:, :, 2] = 1
            self.showKSpace(kSpace)
            print(i)
            vectors = recovery(decayedRotatedMatrix, self.T1, self.TR, self.PD)
            # print(vectors[32,20])

        # kSpace = np.fft.fftshift(kSpace)
        # kSpace = np.fft.fft2(kSpace)
        # for i in range(0, self.phantomSize):
        #     kSpace[i, :] = np.fft.fft(kSpace[i, :])
        # for i in range(0, self.phantomSize):
        #     kSpace[:, i] = np.fft.fft(kSpace[:, i])
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

    def SSFP_reconstruct_image(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        # To be Removed
        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                vectors[i, j] = vectors[i, j].dot(self.PD[i, j])
        print(vectors[30, 10])
        # vectors = rotateX(vectors, self.FA)
        # vectors = recovery(vectors, self.T1, self.TR, self.PD)
        # vectors[:, :, 0] = 0
        # vectors[:, :, 1] = 0
        vectors = self.startup_cycles(vectors)
        vectors = self.TAG_prep(vectors)
        print(vectors[30, 10])

        vectors = rotateX(vectors, self.FA)
        for i in range(0, round(self.phantomSize)):
            rotatedMatrix = vectors
            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, self.TE)

            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            self.showKSpace(kSpace)
            print(i)
            if i % 2 == 0:
                vectors = rotateX(rotatedMatrix, -1 * self.FA * 2)
            else:
                vectors = rotateX(rotatedMatrix, self.FA * 2)

        kSpace = np.fft.fftshift(kSpace)
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

    def SE_image_reconstruct(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        # To be Removed
        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                vectors[i, j] = vectors[i, j].dot(self.PD[i, j])

        vectors = self.startup_cycles(vectors)
        vectors = self.TAG_prep(vectors)

        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                rotatedMatrix = rotateX(vectors, 90)
                rotatedMatrix = gradientXY(rotatedMatrix, stepY, 0)
                decayedMatrix = decay(rotatedMatrix, self.T2, self.TE / 2)
                spinEcho = rotateX(decayedMatrix, 180)
                spinEcho = gradientXY(spinEcho, 0, stepX)
                sigmaX = np.sum(spinEcho[:, :, 0])
                sigmaY = np.sum(spinEcho[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd
            self.showKSpace(kSpace)
            spinEcho[:, :, 0] = 0
            spinEcho[:, :, 1] = 0
            vectors = recovery(spinEcho, self.T1, self.TR, self.PD)

        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

    def TI_Prep(self, vectors):
        vectors = rotateX(vectors, 180)
        vectors = recovery(vectors, self.T1, self.inversion_time, self.PD)
        vectors[:, :, 0] = 0
        vectors[:, :, 1] = 0
        return vectors

    def T2_prep(self, vectors):
        vectors = rotateX(vectors, 90)
        vectors = decay(vectors, self.T2, self.T2_prep_time)
        vectors = rotateX(vectors, -90)
        return vectors

    def TAG_prep(self, vectors):
        t = range(0, self.phantomSize)
        t = np.sin(t)
        for i in range(0, self.phantomSize):
            if (i % self.TAG_frequency == 0):
                for j in range(0, self.phantomSize):
                    vectors[i, j] = np.dot(vectors[i, j], t[j])
        return vectors

    def startup_cycles(self, vectors):
        for i in range(0, self.cycles_number):
            vectors = rotateX(vectors, self.FA)
            vectors = recovery(vectors, self.T1, self.TR, self.PD)
            vectors = decay(vectors, self.T2, self.TR)
            vectors[:, :, 0] = 0
            vectors[:, :, 1] = 0
        return vectors

    def showKSpace(self, img):
        img = img[:]
        # img = np.abs(img)
        img = 20 * np.log(np.abs(img))

        qimg = qimage2ndarray.array2qimage(np.abs(img))
        self.ui.kspaceLbl.setPixmap(QPixmap(qimg))

    def showReconstructedImage(self, img):
        img = img[:]
        img = np.abs(img)
        img = img - np.min(img)
        img = img * (255 / np.max(img))
        img = np.round(img)
        qimg = qimage2ndarray.array2qimage(np.abs(img))
        self.ui.kspaceLbl.setPixmap(QPixmap(qimg))

    def error(self, message):
        errorBox = QMessageBox()
        errorBox.setIcon(QMessageBox.Warning)
        errorBox.setWindowTitle('WARNING')
        errorBox.setText(message)
        errorBox.setStandardButtons(QMessageBox.Ok)
        errorBox.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
