import matplotlib.pyplot as plt
import numpy as np
import math
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow
from rotation import rotateX
from RD import recovery, decay


def ernst_angle(vectors, TR, T1, PD, T2):
    MAX = 0
    matrix = vectors
    angles = range(1,181)
    magnitudes = []

    for i in range(1, 181):
        print(i)
        vectors = startup_cycles(matrix, i, TR, T1, PD, T2)
        sigmaXY = np.sum(np.abs(vectors[:, :, 1]))
        magnitudes.append(sigmaXY)
    plt.plot(angles,magnitudes)
    plt.ylabel("Signal")
    plt.xlabel("Angle in Degree")
    plt.title("POP UP WINDOW GAMDA GEDN")
    plt.show()

def ernst_angle_equation(TR, T1):
    angles = range(1,181)
    mags = np.arccos(np.exp())


def startup_cycles( vectors, FA, TR, T1, PD, T2):
    for i in range(0, 10):
        vectors = rotateX(vectors, FA)
        vectors = recovery(vectors, T1, TR, PD)
        vectors = decay(vectors, T2, TR)
        vectors[:, :, 0] = 0
        #vectors[:, :, 1] = 0
    return vectors
