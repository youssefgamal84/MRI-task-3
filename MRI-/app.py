from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
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
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from math import sin, cos, pi
import csv
from ernst import ernst_angle
from scipy.ndimage.interpolation import shift
import sk_dsp_comm.sigsys as ss  # pip install sk_dsp_comm

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
        self.ui.prepSelc.currentTextChanged.connect(self.showFeatPrep)
        self.ui.Invtime.show()
        self.ui.InvTimeText.show()
        self.ui.spacingSW.hide()
        self.ui.spacingText.hide()
        self.ui.T2prep.hide()
        self.ui.T2prepText.hide()
        self.ui.startSeq.clicked.connect(self.runSequence)
        self.ui.showGraphics.clicked.connect(self.plotGraph)
        self.ui.FlipAngle.textChanged.connect(self.setFA)
        self.ui.TimeEcho.textChanged.connect(self.setTE)
        self.ui.TimeRepeat.textChanged.connect(self.setTR)
        self.ui.btnBrowse.clicked.connect(self.browse)
        self.ui.linkZoom.stateChanged.connect(self.zoomLinkChanged)
        # Mouse Events
        self.ui.phantomlbl.setMouseTracking(False)
        self.ui.kspaceLbl.setMouseTracking(False)

        self.ui.phantomlbl.mouseMoveEvent = self.ui.phantomlbl.editPosition
        self.ui.phantomlbl.wheelEvent = self.ui.phantomlbl.zoomInOut

        self.ui.phantomlbl.mouseDoubleClickEvent = self.pixelClicked
        self.ui.actionDrag_2.triggered.connect(self.selector)
        self.ui.actionBrightness_Contrast.triggered.connect(self.selector2)

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
        self.phantomSize = 512

        self.FA = 90
        self.cosFA = 0
        self.sinFA = 1
        self.TE = 0.001
        self.TR = 0.5
        self.x = 0
        self.y = 0

        # Artifacting
        self.shifting_artifact = True

        self.pixelsClicked = [(0, 0), (0, 0), (0, 0)]
        self.pixelSelector = 0

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
                self.showPhantomImage(self.PD)
            except (IOError, SyntaxError):
                self.error('Check File Extension')

    def zoomLinkChanged(self, state):
        if state:
            self.ui.phantomlbl.wheelEvent = self.linked_zooming
            self.ui.kspaceLbl.wheelEvent = self.linked_zooming
            self.ui.phantomlbl.mouseMoveEvent = self.linked_dragging
            self.ui.kspaceLbl.mouseMoveEvent = self.linked_dragging
            self.ui.kspaceLbl.zoomLevel = self.ui.phantomlbl.zoomLevel
            self.ui.kspaceLbl.rowOffset = self.ui.phantomlbl.rowOffset
            self.ui.kspaceLbl.colOffset = self.ui.phantomlbl.colOffset
        else:
            self.ui.phantomlbl.wheelEvent = self.ui.phantomlbl.zoomInOut
            self.ui.kspaceLbl.wheelEvent = self.ui.kspaceLbl.zoomInOut
            self.ui.phantomlbl.mouseMoveEvent = self.ui.phantomlbl.editPosition
            self.ui.kspaceLbl.mouseMoveEvent = self.ui.kspaceLbl.editPosition

    def linked_zooming(self, event):
        self.ui.phantomlbl.zoomInOut(event)
        self.ui.kspaceLbl.zoomInOut(event)

    def linked_dragging(self, event):
        self.ui.phantomlbl.editPosition(event)
        self.ui.kspaceLbl.editPosition(event)

    def selector(self):
        self.ui.phantomlbl.mouseMoveEvent = self.ui.phantomlbl.editPosition
        self.ui.kspaceLbl.mouseMoveEvent = self.ui.kspaceLbl.editPosition
        self.ui.phantomlbl.setCursor(QCursor(Qt.OpenHandCursor))
        self.ui.kspaceLbl.setCursor(QCursor(Qt.OpenHandCursor))
        print('hi')

    def selector2(self):
        self.ui.phantomlbl.mouseMoveEvent = self.ui.phantomlbl.editBrightnessAndContrast
        self.ui.kspaceLbl.mouseMoveEvent = self.ui.kspaceLbl.editBrightnessAndContrast
        self.ui.phantomlbl.setCursor(QCursor(Qt.SizeVerCursor))
        self.ui.kspaceLbl.setCursor(QCursor(Qt.SizeVerCursor))

    def showPhantom(self, value):
        size = int(value)
        self.phantomSize = size
        self.ui.phantomlbl.phantomSize = size
        self.ui.kspaceLbl.phantomSize = size
        img = phantom(size)
        self.PD = img
        img = img * 255
        self.img = img
        self.ui.phantomlbl.setImg(self.img)
        self.T1 = phantom(size, 'T1')
        self.T2 = phantom(size, 'T2')
        self.originalPhantom = img
        self.pixelsClicked = [(0, 0), [0, 0], [0, 0]]
        self.showPhantomImage(self.img)

    def showPhantomImage(self, img):
        self.ui.phantomlbl.showPhantomImage(img)

    def showFeatPrep(self):
        if self.ui.prepSelc.currentText() == 'T2prep':
            self.ui.Invtime.hide()
            self.ui.InvTimeText.hide()
            self.ui.spacingSW.hide()
            self.ui.spacingText.hide()
            self.ui.T2prep.show()
            self.ui.T2prepText.show()
        if self.ui.prepSelc.currentText() == 'Inversion':
            self.ui.Invtime.show()
            self.ui.InvTimeText.show()
            self.ui.spacingSW.hide()
            self.ui.spacingText.hide()
            self.ui.T2prep.hide()
            self.ui.T2prepText.hide()
        if self.ui.prepSelc.currentText() == 'Tagging':
            self.ui.Invtime.hide()
            self.ui.InvTimeText.hide()
            self.ui.spacingSW.show()
            self.ui.spacingText.show()
            self.ui.T2prep.hide()
            self.ui.T2prepText.hide()

    def changePhantomMode(self, value):

        if value == "PD":
            self.img = self.PD
        if value == "T1":
            self.img = self.T1
        if value == "T2":
            self.img = self.T2

        self.img = self.img * (255 / np.max(self.img))
        self.originalPhantom = self.img

        self.ui.phantomlbl.setImg(self.img)
        self.showPhantomImage(self.img)

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
            x = event.pos().x() * ((self.phantomSize - self.ui.phantomlbl.zoomLevel * 2) / xt)
            y = event.pos().y() * ((self.phantomSize - self.ui.phantomlbl.zoomLevel * 2) / yt)
            x = x + self.ui.phantomlbl.colOffset + self.ui.phantomlbl.zoomLevel
            y = y + self.ui.phantomlbl.rowOffset + self.ui.phantomlbl.zoomLevel
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

    def plotting(self, color, T1=1000, T2=45):
        t1graph = self.ui.graphicsPlotT1
        t2gragh = self.ui.graphicsPlotT2
        # theta = self.FA * pi / 180
        t = np.linspace(0, 10000, 1000)
        t1graph.plot(np.exp(-t / T1) * self.cosFA + 1 - np.exp(-t / T1), pen=pg.mkPen(color))
        t2gragh.plot(self.sinFA * np.exp(-t / T2), pen=pg.mkPen(color))

    #     plotting the graphical representaion of the sequence
    def plotGraph(self):
        self.ui.tabWidget.setCurrentIndex(3)
        self.graphicRep = self.ui.graphicsRep
        self.graphicRep.setRange(xRange=[-10, 50])
        self.graphicRep.setRange(yRange=[0, 8])

        if self.ui.FlipAngle.text() != '':
            FA = int(self.ui.FlipAngle.text())
            FA = FA / 90
            print(FA)
        else:
            FA = 1
            print(FA)

        if self.ui.FlipAngle.text() != '':
            t2 = int(self.ui.T2prep.text())
            print(t2)
        else:
            t2 = 2
            print(t2)

        if self.ui.prepSelc.currentText() == 'Inversion' and self.ui.acqBox.currentText() == 'GRE':
            print(1)
            self.drawRf(2, FA, 0, 0, 10)
            self.drawGz(.5, 0, 0)
            self.drawGy(.5, 0, 0)
            self.drawGx(.5, 0, 0)
        if self.ui.prepSelc.currentText() == 'Inversion' and self.ui.acqBox.currentText() == 'SSFP':
            print(2)
            self.drawRf(2, FA, 0, 0, 10)
            self.drawGz(.5, 0, 0)
            self.drawGy(.5, -.5, 10)
            self.drawGx(.5, -.5, -1, 4)
        if self.ui.prepSelc.currentText() == 'Inversion' and self.ui.acqBox.currentText() == 'SE':
            print(3)
            self.drawRf(2, 2, FA, 0, 20, 5)
            self.drawGz(.5, .5, 3)
            self.drawGy(.5, 0, 0)
            self.drawGx(.5, .5, -6, 4)
        if self.ui.prepSelc.currentText() == 'T2prep' and self.ui.acqBox.currentText() == 'GRE':
            print(4)
            self.drawRf(1, -1, FA, 0, 5, 5, 0, t2)
            self.drawGz(.5, 0, 0)
            self.drawGy(.5, 0, 0)
            self.drawGx(.5, 0, 0)
        if self.ui.prepSelc.currentText() == 'T2prep' and self.ui.acqBox.currentText() == 'SSFP':
            print(5)
            self.drawRf(1, -1, FA, 0, 5, 5, 0, t2)
            self.drawGz(.5, 0, 0)
            self.drawGy(.5, -.5, 10)
            self.drawGx(.5, -.5, -1, 4)
        if self.ui.prepSelc.currentText() == 'T2prep' and self.ui.acqBox.currentText() == 'SE':
            print(6)
            self.drawRf(1, -1, 2, FA, 5, 10, (10 / 3), t2)
            self.drawGz(.5, .5, 3)
            self.drawGy(.5, 0, 0)
            self.drawGx(.5, .5, -6, 4)
        if self.ui.prepSelc.currentText() == 'Tagging' and self.ui.acqBox.currentText() == 'GRE':
            print(7)
            self.drawRf(0, FA, 0, 0, 10)
            self.drawSin()
            self.drawGz(.5, 0, 0)
            self.drawGy(.5, 0, 0)
            self.drawGx(.5, 0, 0)
        if self.ui.prepSelc.currentText() == 'Tagging' and self.ui.acqBox.currentText() == 'SSFP':
            print(8)
            self.drawRf(0, FA, 0, 0, 10)
            self.drawSin()
            self.drawGz(.5, 0, 0)
            self.drawGy(.5, -.5, 10)
            self.drawGx(.5, -.5, -1, 4)
        if self.ui.prepSelc.currentText() == 'Tagging' and self.ui.acqBox.currentText() == 'SE':
            print(9)
            self.drawRf(0, 2, FA, 0, 20, 5)
            self.drawSin()
            self.drawGz(.5, .5, 3)
            self.drawGy(.5, 0, 0)
            self.drawGx(.5, .5, -6, 4)

        ## RF pulse

    def drawGx(self, angle1=0.5, angle2=0.5, dist1=0, dist2=0):
        tx = np.arange(-100, 100, .01)
        x_rect = ss.rect(tx - 20, 5)
        x_rect = x_rect * angle1
        self.graphicRep.plot(tx + dist1, x_rect + 1, pen=pg.mkPen('g'))
        tx = np.arange(-100, 100, .01)
        x_rect = ss.rect(tx - 20, 5)
        x_rect = x_rect * angle2
        if angle2 != 0:
            self.graphicRep.plot(tx + dist2, x_rect + 1, pen=pg.mkPen('g'))

    def drawGy(self, angle1=0.5, angle2=0.5, dist=0):
        ty = np.arange(-100, 100, .01)
        x_rect = ss.rect(ty - 14, 5)
        x_rect = x_rect * angle1
        self.graphicRep.plot(ty, x_rect + 2, pen=pg.mkPen('b'))
        ty = np.arange(-100, 100, .01)
        x_rect = ss.rect(ty - 14, 5)
        x_rect = x_rect * angle2
        if angle2 != 0:
            self.graphicRep.plot(ty + dist, x_rect + 2, pen=pg.mkPen('b'))

    def drawGz(self, angle1=0.5, angle2=0.5, dist=0):
        tz = np.arange(-100, 100, .01)
        x_rect = ss.rect(tz - 8, 5)
        x_rect = x_rect * angle1
        self.graphicRep.plot(tz, x_rect + 3, pen=pg.mkPen('r'))
        tz = np.arange(-50, 100, .01)
        x_rect = ss.rect(tz - 8, 5)
        x_rect = x_rect * angle2
        if angle2 != 0:
            self.graphicRep.plot(tz + dist + 8, x_rect + 3, pen=pg.mkPen('r'))

    def drawRf(self, angle1=1, angle2=1, angle3=1, angle4=1, dist1=0, dist2=0, dist3=0, t2=0):
        self.graphicRep.clear()
        t = np.arange(-50, 100, .01)
        x_tri = ss.tri(t + 2, 0.2)
        x_tri = x_tri * angle1
        self.graphicRep.plot(t - t2, x_tri + 4, pen=pg.mkPen('y'))
        t = np.arange(-50, 100, .01)
        x_tri = ss.tri(t + 2, 0.2)
        x_tri = x_tri * angle2
        self.graphicRep.plot(t + dist1, x_tri + 4, pen=pg.mkPen('y'))
        t = np.arange(-50, 100, .01)
        x_tri = ss.tri(t + 2, 0.2)
        x_tri = x_tri * angle3
        self.graphicRep.plot(t + (2 * dist2), x_tri + 4, pen=pg.mkPen('y'))
        t = np.arange(-50, 100, .01)
        x_tri = ss.tri(t + 2, 0.2)
        x_tri = x_tri * angle4
        self.graphicRep.plot(t + (3 * dist3), x_tri + 4, pen=pg.mkPen('y'))

    def drawSin(self):
        t = np.arange(0, 3, .01)
        x = np.sin(2 * pi * t)
        x = 0.5 * x
        self.graphicRep.plot(t, x + 4, pen=pg.mkPen('y'))

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
            self.ui.kspaceLbl.setCursor(QCursor(Qt.WaitCursor))
            self.ui.tabWidget.setCurrentIndex(1)
            self.ui.kspaceLbl.setImg(None)
            if self.ui.acqBox.currentText() == 'GRE':
                threading.Thread(target=self.GRE_reconstruct_image).start()
            if self.ui.acqBox.currentText() == 'SSFP':
                threading.Thread(target=self.SSFP_reconstruct_image).start()
            if self.ui.acqBox.currentText() == 'SE':
                threading.Thread(target=self.SE_image_reconstruct).start()
            return

    def GRE_reconstruct_image(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        # Proton Density effect
        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                vectors[i, j] = vectors[i, j].dot(self.PD[i, j])

        vectors = self.prepare_vectors(vectors)

        for i in range(0, round(self.phantomSize)):
            if self.shifting_artifact and i == 10:
                self.make_shift(vectors)

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
        if self.shifting_artifact:
            self.T1 = phantom(self.phantomSize,"t1")
            self.T2 = phantom(self.phantomSize,"t2")
        self.ui.kspaceLbl.setCursor(QCursor(Qt.ArrowCursor))

    def make_shift(self, vectors):
        shiftValue = int(self.phantomSize / 4)
        vectors[0:self.phantomSize - shiftValue, :, :], vectors[self.phantomSize - shiftValue:self.phantomSize, :,
                                                        :] = vectors[shiftValue:self.phantomSize, :, :], np.zeros(
            (shiftValue, self.phantomSize, 3))
        self.T1[0:self.phantomSize - shiftValue, :], self.T1[self.phantomSize - shiftValue:self.phantomSize,
                                                     :] = self.T1[shiftValue:self.phantomSize, :], np.zeros(
            (shiftValue, self.phantomSize))
        self.T2[0:self.phantomSize - shiftValue, :], self.T2[self.phantomSize - shiftValue:self.phantomSize,
                                                     :] = self.T2[shiftValue:self.phantomSize, :], np.zeros(
            (shiftValue, self.phantomSize))

    def SSFP_reconstruct_image(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        # Proton Density Effect
        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                vectors[i, j] = vectors[i, j].dot(self.PD[i, j])

        # ernst_angle(vectors, self.TR, self.T1, self.PD, self.T2)
        print(vectors[30, 10])
        # vectors = rotateX(vectors, self.FA)
        # vectors = recovery(vectors, self.T1, self.TR, self.PD)
        # vectors[:, :, 0] = 0
        # vectors[:, :, 1] = 0

        vectors = self.prepare_vectors(vectors)
        print(vectors[30, 10])

        vectors = rotateX(vectors, self.FA)
        for i in range(0, round(self.phantomSize)):
            if self.shifting_artifact and i == 10:
                self.make_shift(vectors)
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
        img = np.fft.fft2(kSpace)
        center = self.phantomSize / 2
        center = int(center)
        img[0:center, :], img[center:self.phantomSize, :] = img[center:self.phantomSize, :], img[0:center, :].copy()
        self.showReconstructedImage(img)
        if self.shifting_artifact:
            self.T1 = phantom(self.phantomSize,"t1")
            self.T2 = phantom(self.phantomSize,"t2")
        self.ui.kspaceLbl.setCursor(QCursor(Qt.ArrowCursor))

    def prepare_vectors(self, vectors):
        vectors = self.startup_cycles(vectors)
        if self.ui.prepSelc.currentText() == 'T2prep':
            vectors = self.T2_prep(vectors)
        elif self.ui.prepSelc.currentText() == 'Tagging':
            vectors = self.TAG_prep(vectors)
        else:
            vectors = self.TI_Prep(vectors)
        return vectors

    def SE_image_reconstruct(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        # To be Removed
        for i in range(0, self.phantomSize):
            for j in range(0, self.phantomSize):
                vectors[i, j] = vectors[i, j].dot(self.PD[i, j])

        vectors = self.prepare_vectors(vectors)

        for i in range(0, self.phantomSize):
            if i == 10 and self.shifting_artifact:
                self.make_shift(vectors)
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

        img = np.fft.fft2(kSpace)
        img = np.fliplr(img)
        self.showReconstructedImage(img)
        if self.shifting_artifact:
            self.T1 = phantom(self.phantomSize,"t1")
            self.T2 = phantom(self.phantomSize,"t2")
        self.ui.kspaceLbl.setCursor(QCursor(Qt.ArrowCursor))

    def TI_Prep(self, vectors):
        vectors = rotateX(vectors, 180)
        if self.ui.Invtime.text() != '':
            self.inversion_time = float(self.ui.Invtime.text())
        vectors = recovery(vectors, self.T1, self.inversion_time, self.PD)
        # vectors[:, :, 0] = 0
        # vectors[:, :, 1] = 0
        return vectors

    def T2_prep(self, vectors):
        vectors = rotateX(vectors, 90)
        if self.ui.T2prep.text() != '':
            self.T2_prep_time = float(self.ui.T2prep.text())
        vectors = decay(vectors, self.T2, self.T2_prep_time)
        vectors = rotateX(vectors, -90)
        # vectors[:, :, 0] = 0
        # vectors[:, :, 1] = 0
        return vectors

    def TAG_prep(self, vectors):
        t = range(0, self.phantomSize)
        t = np.sin(t)
        if self.ui.spacingSW.text() != '':
            self.TAG_frequency = int(self.ui.spacingSW.text())
        for i in range(0, self.phantomSize):
            if (i % self.TAG_frequency == 0):
                for j in range(0, self.phantomSize):
                    vectors[i, j] = np.dot(vectors[i, j], t[j])
        return vectors

    def startup_cycles(self, vectors):
        if self.ui.startCyc.text() != '':
            self.cycles_number = int(self.ui.startCyc.text())
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
        self.ui.kspaceLbl.setImg(img)

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
