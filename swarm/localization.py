#%%
########################################################################
# This code holds classes and methods that recognize and localizes
# robots on the plane.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import sys
import time
from itertools import combinations
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
        self._colors = {'k':(  0,  0,  0),'r':(  0,  0,255),'b':(255,  0,  0),
                        'g':(  0,255,  0),'w':(255,255,255)}
                        #,'m':(255,  0,255),
                        #'y':(  0,255,255),'c':(255,255,  0)}
        self._set_hsv_ranges()
        # Calibration parameters.
        self._img_name_prefix = "cal_img"
        self._img_dir_prefix = "calibration_img"
        self._mtx = None
        self._dist = None
        self._nmtx = None
        self._roi_undistort = None
        # Space boundary parameters.
        self._roi_frame = None
        self._roi_space = None
        self._space_limits_mm = None # (ubx, uby) symmetric space.
        self._center = None          # (center_x, center_y)
        self._mask_space = None      # Mask for work space.
        # Calibration
        if save_image:
            self._save_image()
        else:
            self._calibrate()
            # Pixel to mm conversion factor, found via _find_scale.
            self._p2mm = 0.52845085 # self._find_scale()
            self._get_n_set_space()
    
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
        #self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    
    def _set_hsv_ranges(self):
        """
        Defines HSV color ranges for different colors available in 
        _colors. These ranges are used for localizing robot.
        Depending on the exact color used, ranges should be modified.
        To find appropriate ranges use get_color_ranges method.
        """
        hsv_ranges = {}
        # Black or 'k'
        hsv_ranges['k'] = {'lb': [np.array([  0,  0,  0],dtype=np.uint8)],
                           'ub': [np.array([179,255, 30],dtype=np.uint8)]}
        # Red or 'r'
        hsv_ranges['r'] = {'lb': [np.array([  0, 50, 60],dtype=np.uint8),
                                  np.array([165, 50, 60],dtype=np.uint8)],
                           'ub': [np.array([ 10,255,255],dtype=np.uint8),
                                  np.array([179,255,255],dtype=np.uint8)]}
        # Blue or 'b'
        hsv_ranges['b'] = {'lb': [np.array([116, 60, 31],dtype=np.uint8)],
                           'ub': [np.array([130,255,255],dtype=np.uint8)]}
        # Green or 'g'
        hsv_ranges['g'] = {'lb': [np.array([ 90, 20, 10],dtype=np.uint8)],
                           'ub': [np.array([115,255, 95],dtype=np.uint8)]}
        # White or 'w'
        hsv_ranges['w'] = {'lb': [np.array([ 95, 10,160],dtype=np.uint8)],
                           'ub': [np.array([125, 90,255],dtype=np.uint8)]}
        #
        self._hsv_ranges = hsv_ranges

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
    
    def _undistort(self, img):
        # undistort
        dst = cv2.undistort(img, self._mtx, self._dist, None, self._nmtx)
        return dst
    
    @staticmethod
    def _crop(img, roi):
        """
        Crops image.
        ----------
        Parameters
        ----------
        img: numpy nd.array
            An image array.
        roi: 1D array
            roi = (top_left_x, top_left_y, width, heigth)
        ----------
        Returns
        ----------
        img: numpy nd.array
        """
        x, y, w, h = roi
        return img[y:y+h, x:x+w]

    @staticmethod
    def _find_distance(points):
        """
        Calculates distance of points in a list or 2D numpy array.
        ----------
        Parameters
        ----------
        points: numpy nd.array
            2D array of points.
        ----------
        Returns
        ----------
        distances: numpy nd.array
            Orderred array of distances between all unique pairs.
        """
        distances = []
        # Produces all unique pair of indexes.
        indexes = combinations(range(points.shape[0]),2)
        # Calculate distance for them.
        for index in indexes:
            dist = np.linalg.norm(points[index[0],:] - points[index[1],:])
            distances += [dist]
        return np.array(distances)

    def _find_scale(self):
        """
        Finds "mm/pixel" of the undistorded image.
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
        distances = self._find_distance(objp)*20
        # Arrays to store object points and image points.
        imgpoints = [] # 2d points in image plane.
        mm2pixel = []
        try:
            images = glob.glob(img_path)
            if not len(images):
                raise IOError
            for fname in images:
                # Read image and undistort it.
                img = cv2.imread(fname)
                img = self._undistort(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners=cv2.findChessboardCorners(gray,(n_row,n_col),None)
                # If found, add object points, image points.
                if ret == True:
                    corners2 = cv2.cornerSubPix(gray,corners, (11,11),
                                                (-1,-1), criteria).squeeze()
                    imgpoints.append(corners2)
                    # Calculate scaling.
                    distance_pix = distances/self._find_distance(corners2)
                    mm2pixel += [np.mean(distance_pix)]
        #
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
        return np.mean(mm2pixel)
    
    def _get_n_set_space(self):
        """
        Determines boundaries of rectangular work space in the camera frame.
        Look up open cv python tutorial for details.
        """
        # Our space boundary, edit based on space dimensions.
        W_mm = 240
        H_mm = 190
        offset = 20
        try:
            for _ in range(10):
                # Take each frame
                _, frame = self.cap.read()
            # Undistort
            frame = self._undistort(frame)
            H, W, _ = frame.shape
            # Space boundary is painted black for ease of image processing.
            # Change color space and get a mask for color of space boundary.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = np.zeros(hsv.shape[:2],dtype=np.uint8)
            for lb, ub in zip(self._hsv_ranges['k']['lb'],
                                                  self._hsv_ranges['k']['ub']):
                mask += cv2.inRange(hsv, lb, ub)
            # Find the space boundary ractangle among all contours.
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            msg='Border not found. Work space must be inside camera field of view.'
            if not len(contours):
                raise IOError(msg)
            # Get area of external contours bounding internal contours.
            external_areas = [cv2.contourArea(contours[idx]) if elem >-1 else 0
                                    for idx,elem in enumerate(hierarchy[0,:,2])]
            # Find index of space border.
            space_area = max(external_areas)
            space_border_index = external_areas.index(space_area)
            space_border_index = hierarchy[0,space_border_index,2]
            cnt = contours[space_border_index]
            # If the specified contour area is significantly smaller than 
            # our physical space, workspace is out of camera field of view.
            if space_area < .85*W_mm*H_mm/self._p2mm**2:
                raise IOError(msg)
            # Approximate contour and derive space boundaries.
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            x,y,w,h = cv2.boundingRect(approx)
            # Make w and h odd so we have a definite pixelwise center.
            w = w if w%2 else w-1
            h = h if h%2 else h-1
            # Calculate offset
            left = W - (x+w)
            down = H - (y + h)
            offset = min((x,y, left, down, offset))
            # Calculate ROIs
            roi_frame = (x-offset, y-offset, w+2*offset, h+2*offset)
            roi_space = (offset, offset, w, h)
            center = (int(w/2+offset), int(h/2+offset))
            # Make mask for space.
            mask_space = np.zeros((roi_frame[3], roi_frame[2]), dtype=np.uint8)
            mask_space[offset:offset+h, offset:offset+w] = 255
            #
            self._roi_frame = roi_frame # Used for cropping frames.
            self._roi_space = roi_space
            self._space_limits_mm = ((w-1)*self._p2mm/2, (h-1)*self._p2mm/2)
            self._center = center
            self._mask_space = mask_space
        except Exception as exc:
            print("Ooops! exception happened. Exception details:")
            print(type(exc).__name__,exc.args)
            pass

    def get_color_ranges(self):
        """
        Gives HSV code of the point clicked by mouse.
        Use this function to get appropriate values for _set_hsv_ranges.
        Part of this the function is totally from stackoverflow.
        """
        # Define local event callbacks for mouse.
        # mouse callback function
        def mouse_cb(event,x,y,flags,param):
            nonlocal frame
            # Event happens when left mouse key is released.
            if event == cv2.EVENT_LBUTTONUP:
                # Print HSV color of mouse position.
                bgr = frame[y:y+1,x:x+1]
                hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).squeeze()
                bgr = bgr.squeeze()
                msg = f"Hue: {hsv[0]:3d}, Sat: {hsv[1]:3d}, Val: {hsv[2]:3d}||"
                msg += f"B: {bgr[0]:3d}, G: {bgr[1]:3d}, R: {bgr[2]:3d}."
                print(msg)
        
        def nothing(x):
            pass
        #
        cv2.namedWindow("cam",cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('cam',mouse_cb)
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('H_min', 'cam', 0, 179, nothing)
        cv2.createTrackbar('S_min', 'cam', 0, 255, nothing)
        cv2.createTrackbar('V_min', 'cam', 0, 255, nothing)
        cv2.createTrackbar('H_max', 'cam', 0, 179, nothing)
        cv2.createTrackbar('S_max', 'cam', 0, 255, nothing)
        cv2.createTrackbar('V_max', 'cam', 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv2.setTrackbarPos('H_max', 'cam', 179)
        cv2.setTrackbarPos('S_max', 'cam', 255)
        cv2.setTrackbarPos('V_max', 'cam', 255)

        # Initialize HSV min/max values
        h_min = s_min = v_min = h_max = s_max = v_max = 0
        print('*'*72)
        print('Click to get HSV. Press escape to quit.')
        while True:
            # Read frame
            has_frame, frame = self.cap.read()
            if not has_frame:
                break
            # Get current positions of all trackbars
            h_min = cv2.getTrackbarPos('H_min', 'cam')
            s_min = cv2.getTrackbarPos('S_min', 'cam')
            v_min = cv2.getTrackbarPos('V_min', 'cam')
            h_max = cv2.getTrackbarPos('H_max', 'cam')
            s_max = cv2.getTrackbarPos('S_max', 'cam')
            v_max = cv2.getTrackbarPos('V_max', 'cam')
            # Adjust trackbars if necessary.
            h_min = min(h_min,h_max)
            s_min = min(s_min,s_max)
            v_min = min(v_min,v_max)
            cv2.setTrackbarPos('H_min', 'cam', h_min)
            cv2.setTrackbarPos('S_min', 'cam', s_min)
            cv2.setTrackbarPos('V_min', 'cam', v_min)
            # Set minimum and maximum HSV values to display
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask_inverted = cv2.bitwise_not(mask)
            mask_inverted = cv2.cvtColor(mask_inverted,cv2.COLOR_GRAY2BGR)
            filterred = cv2.bitwise_and(frame, frame, mask=mask)
            filterred = cv2.bitwise_or(filterred,mask_inverted)
            image = np.concatenate((frame, filterred), axis=1)
            cv2.imshow('cam',filterred)
            # Press escape to quit.
            if cv2.waitKey(20) & 0xFF == 27:
                break

########## test section ################################################
if __name__ == '__main__':
    camera = Localization()
    #camera.get_color_ranges()
