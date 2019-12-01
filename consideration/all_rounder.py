# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit

class Mouse:
    def __init__(self):
        pass

    def mouse_callback(self):
        pass

class HSV_supporter:
    def __init__(self):
        self.t_init     = False
        self.MB         = True
        self.opening    = True
        self.closing    = True
        self.ColorErase = False
        self.kernel = np.ones((8,8),np.uint8)

        videofile_path = "20191122/nihongi_f_l1.mp4"

    def t_callback(x):
            pass

    
