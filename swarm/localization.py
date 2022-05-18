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
    def __init__(self, save_image = False):
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
        if save_image:
            self._save_image()

    def _set_camera_settings(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
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
        finally:
            self.cap.release()
            cv2.destroyWindow("img")
########## test section ################################################
if __name__ == '__main__':
    camera = Localization()
