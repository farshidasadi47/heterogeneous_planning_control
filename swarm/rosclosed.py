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
