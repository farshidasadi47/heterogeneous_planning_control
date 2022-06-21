#%%
########################################################################
# This module is responsible for classes and methods that publish
# and subscribe to the arduino for closed loop planning  and control.
# For camera communication, data pipeline is used.
# Camera should be installed and working for using this class.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import sys
import time
import re

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Empty
from turtlesim.action import RotateAbsolute

try:
    from swarm import closedloop, model, localization
except ModuleNotFoundError:
    import closedloop, model, localization
########## Definiitons #################################################
class NodeTemplate(Node):
    """Parent class for all nodes used in the project."""
    def __init__(self, name):
        if rclpy.ok() is not True: rclpy.init(args = sys.argv)
        super().__init__(name)
        self.counter = 0
        # QoS profile
        self.qos_profile = QoSProfile(
        reliability=
                   QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        history = QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, depth=1)
        self.pubs_msgs = dict()
        self.subs_msgs = dict()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True: rclpy.shutdown()
        # Catch exceptions
        if exc_type is not KeyboardInterrupt:
            print(type(exc).__name__,exc.args)
        if exc_type is KeyboardInterrupt:
            print("Interrupted by user.")
        return True
    
    def _add_publisher(self, msg_type, topic: str, reliable = False):
        qos_profile = 1 if reliable else self.qos_profile
        self.create_publisher(msg_type,topic,qos_profile)
    
    def _add_subscriber(self, msg_type, topic:str, callback, reliable = False):
        qos_profile = 1 if reliable else self.qos_profile
        self.create_subscription(msg_type,topic,callback,qos_profile)
    
    def _construct_pubsub_msgs(self):
        """Constructs a dictionaries containing pubs and subs msgs."""
        # Get all publishers and subscribers.
        self.pubs_dict = {pub.topic_name:pub for pub in self.publishers}
        self.subs_dict = {sub.topic_name:sub for sub in self.subscriptions}
        # Publishers
        for k, v in self.pubs_dict.items():
            self.pubs_msgs[k] = v.msg_type()  # v is a publisher.
        # Subscribers
        for k, v in self.subs_dict.items():
            self.subs_msgs[k] = v.msg_type()  # v is a subscriber.

    def _add_service_server(self, srv_type, srv_name: str, callback):
        """Adds an action server and puts it into a dictionary."""
        self.create_service(srv_type, srv_name, callback)

    def _add_action_server(self, action_type, action_name, callback):
        ActionServer(self, action_type, action_name, callback)

class MultiThreadedExecutorTemplate(rclpy.executors.MultiThreadedExecutor):
    """Parent class for all multithreaded executors in the project."""
    def __init__(self):
        if rclpy.ok() is not True: rclpy.init(args = sys.argv)
        super().__init__()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True: rclpy.shutdown()
        # Catch exceptions
        if exc_type is not KeyboardInterrupt:
            print(type(exc).__name__,exc.args)
        if exc_type is KeyboardInterrupt:
            print("Interrupted by user.")
        return True

class SingleThreadedExecutorTemplate(rclpy.executors.SingleThreadedExecutor):
    """Parent class for all singlethreaded executors in the project."""
    def __init__(self):
        if rclpy.ok() is not True: rclpy.init(args = sys.argv)
        super().__init__()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True: rclpy.shutdown()
        # Catch exceptions
        if exc_type is not KeyboardInterrupt:
            print(type(exc).__name__,exc.args)
        if exc_type is KeyboardInterrupt:
            print("Interrupted by user.")
        return True

class GetVideo(NodeTemplate):
    """Main node for getting camera frame."""
    def __init__(self, rate = 55, name="getvideo"):
        super().__init__(name)
        self.camera = localization.Localization()
        _, self.frame, _ = self.camera.get_frame(True,True)
        self.br = CvBridge()
        self.current = 0
        self.past = time.time()
        # Add publishers.
        self._add_publisher(Image, "/camera", reliable = False)
        self._add_publisher(Float32MultiArray,"/state_fb", reliable = False)
        # Add timer
        self.timer = self.create_timer(1/rate, self._timer_callback)
        self.timer_cam = self.create_timer(1/30, self._timer_cam_callback)
        # Custruct publisher and subscription instance variables.
        self._construct_pubsub_msgs()

    def _timer_callback(self):
        """Orders a state_fb publication."""
        ret, self.frame, state_fb = self.camera.get_frame(True, True)
        if ret:
            self.pubs_msgs["/state_fb"].data = state_fb.tolist()
            self.pubs_dict["/state_fb"].publish(self.pubs_msgs["/state_fb"])
            self.current = time.time()
            elapsed = round((self.current - self.past)*1000)
            self.past = self.current
            #
            msg=f"{self.current%1e3:+08.3f}|{elapsed:03d}|{self.counter:06d}|"
            none = "None"
            for i, k in enumerate(self.camera._hsv_ranges):
                v = state_fb[3*i:3*i+3]
                if 999 in v:
                    msg += f"'{k:1s}': {none:>21s},"
                else:
                    msg += (f"'{k:1s}': {v[0]:+06.1f},{v[1]:+06.1f},"
                                    f"{np.rad2deg(v[-1]):+07.2f}|")
            print(msg)
        self.counter = (self.counter + 1)%1000000
    
    def _timer_cam_callback(self):
        self.pubs_dict["/camera"].publish(self.br.cv2_to_imgmsg(self.frame))

class ShowVideo(NodeTemplate):
    """Main node for showing cmera frames live."""
    def __init__(self, rate = 30, name="showvideo"):
        super().__init__(name)
        self.window = cv2.namedWindow('workspace',cv2.WINDOW_AUTOSIZE)
        # Some global variable
        self.br = CvBridge()
        self.current = 0
        self.past = time.time()
        # Add subscribers.
        self._add_subscriber(Float32MultiArray,"/state_fb",
                                            self._state_fb_cb, reliable= False)
        self._add_subscriber(Image,"/camera",self._camera_cb, reliable= False)
        # Add timer.
        self.timer = self.create_timer(1/rate, self._timer_callback)
        # Custruct publisher and subscription instance variables.
        self._construct_pubsub_msgs()
    
    def _state_fb_cb(self,msg):
        self.subs_msgs['/state_fb'].data = msg.data

    def _camera_cb(self, msg):
        self.subs_msgs["/camera"] = msg
        self.current = time.time()
        elapsed = round((self.current - self.past)*1000)
        self.past = self.current
        msg=f"vid: {self.current%1e3:+08.3f}|{elapsed:03d}|{self.counter:06d}|"
        self.counter = (self.counter + 1)%1000000
        print(msg)

    def get_subs_values(self):
        """Return the current value of subscribers as an array."""
        try:
            frame = self.br.imgmsg_to_cv2(self.subs_msgs["/camera"])
        except CvBridgeError:
            frame = np.zeros((1,1),dtype=np.uint8)
        return frame
    
    def _timer_callback(self):
        img = self.get_subs_values()
        cv2.imshow('workspace',img)
        cv2.waitKey(1)

class GetVideoExecutor(MultiThreadedExecutorTemplate):
    """Main executor for getting camera frames and feedback."""
    def __init__(self, rate = 55):
        super().__init__()
        self.add_node(GetVideo(rate))
        print("*"*72 + "\nGetVideo node is initialized.\n" + "*"*72)

class ShowVideoExecutor(SingleThreadedExecutorTemplate):
    def __init__(self, rate = 30):
        super().__init__()
        self.add_node(ShowVideo(rate))
        print("*"*72 + "\nShowVideo  node is initialized.\n" + "*"*72)

def get_video():
    with GetVideoExecutor(55) as executor:
        executor.spin()

def show_video():
    with ShowVideoExecutor(30) as video:
        video.spin()
########## Test section ################################################
if __name__ == "__main__":
    get_video()
