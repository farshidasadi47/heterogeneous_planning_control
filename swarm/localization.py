#%%
########################################################################
# This code holds classes and methods that recognize and localizes
# robots on the plane.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2

########## classes and functions #######################################
class Localization():
    """
    This class holds methods to localize robots on the workspace.
    To find robots, different colors are used for them. Camera frames
    are filterred based on specific color ranges to localize robots.
    """
    def __init__(self, save_image = False):
        self.cap = cv2.VideoCapture(-1)
        if not self.cap.isOpened():
            print("Cannot open camera.")
            sys.exit()
        self._W = 640
        self._H = 480
        self._set_camera_settings()
        # Calibration parameters.
        self._img_name_prefix = "cal_img"
        self._img_dir_prefix = "calibration_img"
        self._mtx = None
        self._dist = None
        self._nmtx = None
        self._roi_undistort = None
        # Calibration
        if save_image:
            self._save_image()
        else:
            self._calibrate()
    
    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        # Catch exceptions
        if exc_type == KeyboardInterrupt:
            print("Interrupted by user.")
            return True

    def _set_camera_settings(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    def _save_image(self):
        """
        Shows camera live picture. User can press \"s\" to save the
        current frame. Pressing \"Escape\" quits the function.
        This function can be used to take frames for calibration.
        """
        # Create object points.
        img_name_prefix = self._img_name_prefix
        img_dir_prefix = self._img_dir_prefix
        try:
            counter = 0
            alive = True
            save_img = False
            while alive:
                has_frame, frame = self.cap.read()
                if not has_frame:
                    break
                img = frame
                cv2.imshow('img', img)
                # Save image if requested.
                print_str = f"{time.time()%1e4:+010.3f}|{counter:+010d}|"
                print(print_str)
                counter += 1
                if save_img:
                    print("Saving current frame.")
                    # Set file name for saving animation.
                    img_index = 1
                    img_name = f"{img_name_prefix}_{img_index:02d}.jpg"
                    img_directory = os.path.join(os.getcwd(),img_dir_prefix)
                    # If the directory does not exist, make one.
                    if not os.path.exists(img_directory):
                        os.mkdir(img_directory)
                    img_path = os.path.join(img_directory,img_name)
                    # Check if the current file name exists in the directory.
                    while os.path.exists(img_path):
                        # Increase file number index until no file with such
                        # name exists.
                        img_index += 1
                        img_name = f"{img_name_prefix}_{img_index:02d}.jpg"
                        img_path = os.path.join(img_directory,img_name)
                    # Save the image.
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(img_path,img)
                    save_img = False
                #
                key = cv2.waitKey(1)
                if key == 27:
                    alive = False
                elif key == ord('S') or key == ord('s'):
                    save_img = True
        except KeyboardInterrupt:
            print("Interrupted by user.")
            pass
        except Exception as exc:
            print("Ooops! exception happened.")
            print("Exception details:")
            print(type(exc).__name__,exc.args)
            pass
    
    def _calibrate(self):
        """
        Uses a chess board to calibrate the camera and remove distortion.
        """
        n_row, n_col = 6, 7  # Change this based on your chessboard.
        # First try to read calibration image files.
        img_dir_prefix = self._img_dir_prefix
        img_name_prefix = self._img_name_prefix
        img_directory = os.path.join(os.getcwd(),img_dir_prefix)
        img_path = os.path.join(img_directory,"*.jpg")
        # termination criteria
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), ...,(6,5,0)
        objp = np.zeros((n_row*n_col,3), np.float32)
        objp[:,:2] = np.mgrid[0:n_row,0:n_col].T.reshape(-1,2)
        # Arrays to store object points and image points.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        try:
            images = glob.glob(img_path)
            if not len(images):
                raise IOError
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners=cv2.findChessboardCorners(gray,(n_row,n_col),None)
                # If found, add object points, image points.
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray,corners, (11,11),
                                                             (-1,-1), criteria)
                    imgpoints.append(corners2)
            # Get calibration parameters.
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                            imgpoints,
                                                            gray.shape[::-1],
                                                            None, None)
            self._mtx = mtx
            self._dist = dist
            h, w = self._H, self._W
            nmtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h), 1, (w,h))
            self._nmtx = nmtx
            self._roi_undistort = roi

        except IOError:
            print("Ooops! calibration images are not found in:")
            print(os.path.join(".",img_dir_prefix,img_name_prefix,"ij.jpg"))
            print("Initialize class with \"save_image = True\".")
        except KeyboardInterrupt:
            print("Interrupted by user.")
            pass
        except Exception as exc:
            print("Ooops! exception happened.")
            print("Exception details:")
            print(type(exc).__name__,exc.args)
            pass
    
    def _undistort(self, img, crop = True):
        # undistort
        dst = cv2.undistort(img, self._mtx, self._dist, None, self._nmtx)
        # crop the image
        if crop:
            x, y, w, h = self._roi
            dst = dst[y:y+h, x:x+w]
        return dst
########## test section ################################################
if __name__ == '__main__':
    camera = Localization()
