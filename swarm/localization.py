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

try:
    from swarm import model
except ModuleNotFoundError:
    import model
########## classes and functions #######################################
class Localization():
    """
    This class holds methods to localize robots on the workspace.
    To find robots, different colors are used for them. Camera frames
    are filterred based on specific color ranges to localize robots.
    """
    def __init__(self, raw = False):
        self.cap = cv2.VideoCapture(-1)
        if not self.cap.isOpened():
            print("Cannot open camera.")
            sys.exit()
        self._W = 640
        self._H = 480
        self._fps = 60
        self._fmtcap = 'YUYV'
        self._set_camera_settings()
        model.define_colors(self)
        self._set_hsv_ranges()
        # Calibration parameters.
        self._img_name_prefix = "cal_img"
        # Change based on your real package directory.
        package_dir = r"/home/fasadi/ws/src/swarm/swarm"
        self._img_dir_prefix = os.path.join(package_dir,"calibration_img")
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
        if raw:
            pass
        else:
            # New camera does not have image distortion.
            #self._calibrate() 
            # Pixel to mm conversion factor, found via _find_scale.
            self._p2mm = 0.48970705 #self._find_scale()
            self._get_n_set_space()
            print("Space set.")
    
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
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOCUS, 260)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._H)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*self._fmtcap))
    
    def _set_hsv_ranges(self):
        """
        Defines HSV color ranges for different colors available in 
        _colors. These ranges are used for localizing robot.
        Depending on the exact color used, ranges should be modified.
        To find appropriate ranges use get_color_ranges method.
        """
        hsv_ranges = {}
        if self._fps == 60:
            # For raw format.
            # Black or 'k'
            hsv_ranges['k'] = {'lb': [np.array([  0,  0,  0],dtype=np.uint8)],
                               'ub': [np.array([179,255, 50],dtype=np.uint8)]}
            # Red or 'r'
            hsv_ranges['r'] = {'lb': [np.array([  0,120, 40],dtype=np.uint8),
                                      np.array([160,120, 40],dtype=np.uint8)],
                               'ub': [np.array([ 20,255,255],dtype=np.uint8),
                                      np.array([179,255,255],dtype=np.uint8)]}
            # Blue or 'b'
            hsv_ranges['b'] = {'lb': [np.array([110, 30, 35],dtype=np.uint8)],
                               'ub': [np.array([130,255,255],dtype=np.uint8)]}
            # Green or 'g'
            hsv_ranges['g'] = {'lb': [np.array([ 70, 50,  0],dtype=np.uint8)],
                               'ub': [np.array([ 95,255,160],dtype=np.uint8)]}
            # White or 'w'
            hsv_ranges['w'] = {'lb': [np.array([ 90, 20, 90],dtype=np.uint8)],
                               'ub': [np.array([120,130,255],dtype=np.uint8)]}
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
        img_directory = self._img_dir_prefix
        try:
            counter = 0
            save_img = False
            while True:
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
                    cv2.destroyAllWindows()
                    break
                elif key == ord('S') or key == ord('s'):
                    save_img = True
        except KeyboardInterrupt:
            print("Interrupted by user.")
            pass
        except Exception as exc:
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
        img_path = os.path.join(img_dir_prefix,img_name_prefix + r"_??.jpg")
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
                    # Draw and display the corners
                    #cv2.drawChessboardCorners(img,(n_row,n_col),corners2, ret)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
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
            print(img_path)
            print("Initialize class with \"save_image = True\".")
        except KeyboardInterrupt:
            print("Interrupted by user.")
            pass
        except Exception as exc:
            print(type(exc).__name__,exc.args)
            pass
        finally:
            cv2.destroyAllWindows()
    
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
                #img = self._undistort(img)
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
            for _ in range(60):
                # Take each frame
                _, frame = self.cap.read()
            # Undistort
            #frame = self._undistort(frame)
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
            if not len(contours):
                raise IOError
            # Get area of external contours bounding internal contours.
            external_areas = [cv2.contourArea(contours[idx]) if elem >-1 else 0
                                    for idx,elem in enumerate(hierarchy[0,:,2])]
            # Find index of external border.
            external_border_index = external_areas.index(max(external_areas))
            # Workspace is the biggest internal contour.
            internal_areas = [cv2.contourArea(contours[idx])
                                if elem == external_border_index else 0
                                for idx, elem in enumerate(hierarchy[0,:,3])]
            space_area = max(internal_areas)
            space_border_index = internal_areas.index(space_area)
            cnt = contours[space_border_index]
            # If the specified contour area is significantly smaller than 
            # our physical space, workspace is out of camera field of view.
            if space_area < .85*W_mm*H_mm/self._p2mm**2:
                raise IOError
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
            roi_space = (offset + 6, offset + 6, w - 12, h - 12)
            center = (int(w/2+offset), int(h/2+offset))
            # Make mask for space.
            mask_space = np.zeros((roi_frame[3], roi_frame[2]), dtype=np.uint8)
            mask_space[roi_space[1]:roi_space[1]+roi_space[3],
                       roi_space[0]:roi_space[0]+roi_space[2]] = 255
            #
            self._roi_frame = roi_frame # Used for cropping frames.
            self._roi_space = roi_space
            self._space_limits_mm = ((w-1)*self._p2mm/2, (h-1)*self._p2mm/2)
            self._center = center
            self._mask_space = mask_space
        except IOError:
            print('Border not found.')
            pass
        except Exception as exc:
            print(type(exc).__name__,exc.args)
            pass
    
    @staticmethod
    def _pixel_state_from_rotated_rect(rect):
        """
        Retrieve position and angle of robot from its cv2.RotatedRect.
        ----------
        Parameters
        ----------
        rect: cv2.RotatedRect object tuple.
        ----------
        Returns
        ----------
        pixel_state: numpy nd.array
            pixel_state = [x_pixel, y_pixel, angle]
        """
        pixel_state = []
        pixel_state.extend(rect[0])
        pixel_state.append(np.deg2rad(rect[2]))
        if rect[1][0] > rect[1][1]:
            # Since we want the angle of the normal to longer side.
            pixel_state[2] = pixel_state[2]- np.pi/2
        return pixel_state

    def _find_robot(self, hsv, color):
        """
        Finds and returns position and angle of robot with specific 
        color in the given hsv frame.
        """
        real_robo_area = 4*10/self._p2mm**2
        mask = np.zeros(hsv.shape[:2],dtype=np.uint8)
        for lb, ub in zip(self._hsv_ranges[color]['lb'],
                                                self._hsv_ranges[color]['ub']):
            mask += cv2.inRange(hsv, lb, ub)
        # Find the external contours in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if not len(contours):
            return None
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        #areas = np.ma.masked_less(areas, .25*real_robo_area)
        # Filter contours with areas outside of expected range.
        areas=np.ma.masked_outside(areas,.4*real_robo_area,2*real_robo_area)
        if areas.count() == 0:
            return None
        # Find the nearest area to size of our physical robots
        idx = (np.abs(areas - real_robo_area)).argmin()
        cnt = contours[idx]
        rect = cv2.minAreaRect(cnt)
        pixel_state = self._pixel_state_from_rotated_rect(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return pixel_state, box

    def _pixel2cartesian(self, point):
        """
        Converts pixel coordinate and angle to cartesian equivalent.
        ----------
        Parameters
        ----------
        point: 1D array
            point = [x_pixel, y_pixel, angle: optional]
        ----------
        Returns
        ----------
        cartesian: 1D array
            cartesian = [x, y, angle: optional]
        """
        cartesian = np.zeros_like(point,dtype=float)
        cartesian[-1] = -point[-1]                             # Angle
        cartesian[0] = (point[0] - self._center[0])*self._p2mm # x
        cartesian[1] = (self._center[1] - point[1])*self._p2mm # y
        return cartesian
    
    def _cartesian2pixel(self,point):
        """Does opposite of _pixel2cartesian."""
        pixel = np.zeros(2,dtype=int)
        pixel[0] = int(point[0]/self._p2mm) + self._center[0]
        pixel[1] = self._center[1] - int(point[1]/self._p2mm)
        return pixel
    
    def _draw_grid(self,frame, x_spacing = 20, y_spacing = 20):
        """Draws grid lines on a given frame with specified spacing."""
        color = (99,180,40) # kind of green
        ubx, uby = self._space_limits_mm
        lbx, lby = -ubx, -uby
        _, _, w, h = self._roi_frame
        # Origin
        frame = cv2.circle(frame, self._center, 10, color, 1,cv2.LINE_AA)
        frame = cv2.ellipse(frame, self._center,(10,10), 
                                                 0,90,180,color,-1,cv2.LINE_AA)
        frame = cv2.ellipse(frame, self._center,(10,10), 
                                                 0,270,360,color,-1,cv2.LINE_AA)                                        
        # Vertical lines.
        vertical_r = np.arange(0.0,ubx, x_spacing)
        vertical_l = np.arange(-x_spacing,lbx,-x_spacing)
        vertical = np.concatenate((vertical_l, vertical_r))
        for vert in vertical:
            pixel = self._cartesian2pixel([vert,0.0])
            frame =cv2.line(frame,(pixel[0],0),(pixel[0],h),color,1,cv2.LINE_8)
        # Horizontal lines.
        horizontal_u = np.arange(0.0,uby, y_spacing)
        horizontal_d = np.arange(-y_spacing,lby,-y_spacing)
        horizontal = np.concatenate((horizontal_d, horizontal_u))
        for horz in horizontal:
            pixel = self._cartesian2pixel([0.0,horz])
            frame =cv2.line(frame,(0,pixel[1]),(w,pixel[1]),color,1,cv2.LINE_8)
        return frame

    def get_frame(self, draw_robots = True, draw_grid= False):
        """
        Gets one frame from camera, process it and returns the frame.
        """
        # Take frame.
        ret, frame = self.cap.read()
        if ret:
            # Process frame.
            #frame = self._undistort(frame)
            frame = self._crop(frame,self._roi_frame)
            masked = cv2.bitwise_and(frame,frame, mask=self._mask_space)
            hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
            # Draw grid if requested.
            if draw_grid:
                frame = self._draw_grid(frame)
            # Find robots, if any and process them.
            robots = {color: self._find_robot(hsv,color)
                                          for color in self._hsv_ranges.keys()}
            robot_states = np.zeros(len(self._hsv_ranges)*3,dtype=float)
            for i,(k, v) in enumerate(robots.items()):
                if v is None:
                    current_state = np.ones(3,dtype=float)*999.0
                    robot_states[3*i:3*i+3] = current_state
                    continue
                pixel_state, box = v
                current_state = self._pixel2cartesian(pixel_state)
                robot_states[3*i:3*i+3] = current_state
                if draw_robots:
                    cv2.drawContours(frame,[box],-1,self._colors[k],2)
            # Draw space borders.
            x,y,w,h = self._roi_space
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
            return ret, frame, robot_states
        else:
            return ret, frame, None
    
    def stream_test(self,draw_robots = True, draw_grid = True):
        """Streams camera video with requested options."""
        cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE)#WINDOW_KEEPRATIO)
        then = 0
        counter = 1
        while(1):
            # Take each frame
            _, frame, robot_states = self.get_frame(draw_robots, draw_grid)
            cv2.imshow('frame',frame)
            now = time.time()
            freq = 1/(now - then)
            then = now
            print_str = f"{now%1e4:+010.3f}|{freq:010.3f}|{counter:+06d}||"
            rob_str = ""
            none = 'None'
            for i, k in enumerate(self._hsv_ranges):
                v = robot_states[3*i:3*i+3]
                if 999 in v:
                    rob_str += f"'{k:1s}': {none:>21s},"
                else:
                    rob_str += (f"'{k:1s}': {v[0]:+06.1f},{v[1]:+06.1f},"
                                 f"{np.rad2deg(v[-1]):+07.2f}|")
            print(print_str + rob_str[:-1])
            counter += 1
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
    
    def stream_with_choice(self):
        """
        Main camera loop that streams camera data.
        """
        try:
            counter = 0
            then = time.time()
            undistort = False
            cv2.namedWindow("cam",cv2.WINDOW_KEEPRATIO)
            while True:
                has_frame, frame = self.cap.read()
                if not has_frame:
                    break
                cv2.imshow("cam", frame)
                if undistort:
                    #frame = self._undistort(frame)
                    pass
                now = time.time()
                freq = 1/(now - then)
                then = now
                msg = f"{now%1e4:+010.3f}|{freq:010.3f}|{counter:+06d}"
                print(msg)
                counter += 1
                # Read the key
                key = cv2.waitKey(10)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key == ord('U') or key == ord('u'):
                    undistort = True
                elif key == ord('P') or key == ord('p'):
                    undistort = False
        except KeyboardInterrupt:
            print("Interrupted by user.")
            pass
        except Exception as exc:
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
        for _ in range(60):
            has_frame, frame = self.cap.read()
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
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break

########## test section ################################################
if __name__ == '__main__':
    camera = Localization(raw = False)
    camera.stream_test()
    camera.get_color_ranges()
    #camera.stream_with_choice()
    #print(camera._find_scale())
