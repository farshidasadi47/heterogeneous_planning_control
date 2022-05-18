#%%
########################################################################
# This code holds classes and methods that recognize and localizes
# robots on the plane.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import time
import glob

import numpy as np
import cv2

########## classes and functions #######################################
class Localization():
    """
    This class holds methods to localize robots on the plane.
    """
    def __init__(self):
        self.camera_matrix = None
        self.cap = cv2.VideoCapture(0)
        self._h = 480
        self._w = 640
        self._set_camera_settings()
        self._img_name_prefix = "cal_img"
        self._img_dir_prefix = "calibration_img"
        self._mtx = None
        self._dist = None
        self._nmtx = None
        self._roi = None

    def _set_camera_settings(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
########## test section ################################################
if __name__ == '__main__':
    camera = Localization()
