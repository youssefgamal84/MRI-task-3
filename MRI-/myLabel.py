from PyQt5 import QtWidgets
from PyQt5 import QtWidgets, QtCore, QtGui
import math
import qimage2ndarray
import numpy as np
from PyQt5.QtGui import QPixmap

MAX_CONTRAST = 2
MIN_CONTRAST = 0.1
MAX_BRIGHTNESS = 100
MIN_BRIGHTNESS = -100
SAFETY_MARGIN = 10
MAX_PIXELS_CLICKED = 3


class myLabel(QtWidgets.QLabel):

    def __init__(self, parent=None):
        super(myLabel, self).__init__(parent=parent)

        # Parameters for the pixel painting
        self.pixelsClicked = [(0, 0), (0, 0), (0, 0)]
        self.pixelSelector = 0
        self.phantomSize = 32

        self.img = None
        self.originalPhantom = None

        self.wheelEvent = self.zoomInOut
        self.mouseMoveEvent = self.editPosition

        # Parameters for zooming
        self.zoomLevel = 0
        self.rowOffset = 0
        self.colOffset = 0  # to control the position of a Zoomed In picture

        # For Mouse moving, changing Brightness and Contrast
        self.lastY = None
        self.lastX = None

        # For Contrast Control
        self.contrast = 1.0
        self.brightness = 0

    def paintEvent(self, e):
        super().paintEvent(e)
        paint = QtGui.QPainter(self)
        paint.begin(self)

        for pixelSet in self.pixelsClicked:
            x = pixelSet[0]
            y = pixelSet[1]
            if x < self.phantomSize / 2:
                x = x - self.zoomLevel
            else:
                x = x - self.zoomLevel
            if y < self.phantomSize / 2:
                y = y - self.zoomLevel
            else:
                y = y - self.zoomLevel
            x = x - self.colOffset
            y = y - self.rowOffset
            xt = self.frameGeometry().width()
            yt = self.frameGeometry().height()
            x = x * (xt / (self.phantomSize - self.zoomLevel * 2))
            y = y * (yt / (self.phantomSize - self.zoomLevel * 2))
            x = math.ceil(x)
            y = math.ceil(y)
            if self.pixelSelector == 0:
                pen = QtGui.QPen(QtCore.Qt.red)
            if self.pixelSelector == 1:
                pen = QtGui.QPen(QtCore.Qt.blue)
            if self.pixelSelector == 2:
                pen = QtGui.QPen(QtCore.Qt.yellow)
            pen.setWidth(1)
            paint.setPen(pen)
            # draw rectangle on painter
            paint.drawRect(x - 10, y - 10, 20, 20)

            self.pixelSelector += 1
            self.pixelSelector = self.pixelSelector % 3
        paint.end()

    def zoomInOut(self, event):
        direction = event.angleDelta().y() > 0
        if direction:
            self.zoomLevel = self.zoomLevel + 1
        else:
            self.zoomLevel = self.zoomLevel - 1
        # constraints
        if self.zoomLevel < 0:
            self.zoomLevel = 0
            self.rowOffset = 0
            self.colOffset = 0
        elif self.zoomLevel > self.phantomSize / 2 - 2:
            self.zoomLevel = int(self.phantomSize / 2 - 2)

        self.offsetCorrection()

        img = self.img[0 + self.zoomLevel + self.rowOffset:self.phantomSize - self.zoomLevel + self.rowOffset,
              0 + self.zoomLevel + self.colOffset:self.phantomSize - self.zoomLevel + self.colOffset]
        self.showPhantomImage(img)

    def offsetCorrection(self):
        # Sanity Check
        if self.rowOffset > self.zoomLevel:
            self.rowOffset = self.zoomLevel
        if self.colOffset > self.zoomLevel:
            self.colOffset = self.zoomLevel
        if self.rowOffset < -self.zoomLevel:
            self.rowOffset = -self.zoomLevel
        if self.colOffset < -self.zoomLevel:
            self.colOffset = -1 * self.zoomLevel

    def showPhantomImage(self, img):
        self.qimg = qimage2ndarray.array2qimage(img)
        # self.ui.phantomlbl.setAlignment(QtCore.Qt.AlignCenter)
        # self.ui.phantomlbl.setFixedWidth(self.phantomSize)
        # self.ui.phantomlbl.setFixedHeight(self.phantomSize)
        self.setPixmap(QPixmap(self.qimg))

    def setImg(self, img):
        self.img = img
        self.originalPhantom = img

    def editPosition(self, event):
        if self.lastX is None:
            self.lastX = event.pos().x()
        if self.lastY is None:
            self.lastY = event.pos().y()
            return

        currentPositionX = event.pos().x()
        if currentPositionX > self.lastX:
            self.colOffset -= 1
        elif currentPositionX < self.lastX:
            self.colOffset += 1

        currentPositionY = event.pos().y()
        if currentPositionY > self.lastY:
            self.rowOffset -= 1
        elif currentPositionY < self.lastY:
            self.rowOffset += 1

        self.offsetCorrection()

        img = self.img[0 + self.zoomLevel + self.rowOffset:self.phantomSize - self.zoomLevel + self.rowOffset,
              0 + self.zoomLevel + self.colOffset:self.phantomSize - self.zoomLevel + self.colOffset]
        self.showPhantomImage(img)

        self.lastY = currentPositionY
        self.lastX = currentPositionX

    def editBrightnessAndContrast(self, event):
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
        img = self.img[0 + self.zoomLevel:-self.zoomLevel - 1, 0 + self.zoomLevel:-self.zoomLevel - 1]
        self.showPhantomImage(img)

        self.lastY = currentPositionY
        self.lastX = currentPositionX
