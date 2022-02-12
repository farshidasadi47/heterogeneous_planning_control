#%%
########################################################################
# This module is responsible for classes and methods that publish
# and subscribe to the peripherals (arduino and probably camera).
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Point32

from controller import Pipeline
########## Definiitons #################################################
class Peripherals(Node):
    """Main executor for arduino comunications."""
    def __init__(self, pipeline: Pipeline, rate = 100, name="peripherals"):
        # If rclpy.init() is not called, call it.
        if rclpy.ok() is not True:
            rclpy.init(args = sys.argv)
        super().__init__(name)
        # Some global variable
        self.counter = 0
        self.pipeline = pipeline
        # QoS profile
        self.qos_profile = QoSProfile(
        reliability=
                   QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        history = QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        depth=1)
        #self.qos_profile = rclpy.qos.qos_profile_system_default.reliability
        # Add arduino publishers.
        self.__add_publisher(Point32,"/arduino_field_cmd")
        # Order of adding subscriber and times matters.
        # Add arduino subscribers.
        self.__add_subscriber(Point32,"/arduino_field_fb",
                                                    self.__arduino_field_fb_cb)
        # Add timer
        self.timer = self.create_timer(1/rate, self.__timer_callback)
        # Get all publishers and subscribers.
        self.pubs_dict = {pub.topic_name:pub for pub in self.publishers}
        self.pubs_dict.pop('/parameter_events')
        self.subs_dict = {sub.topic_name:sub for sub in self.subscriptions}
        # Cusdtruct publisher and subscription instance variables.
        self.pubs_msgs = dict()
        self.subs_msgs = dict()
        self.__construct_msgs()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True:
            rclpy.shutdown()
        # Catch exceptions
        if exc_type == KeyboardInterrupt:
            print("Interrupted by user.")
            return True
    
    def __add_publisher(self, msg_type, topic: str):
        """Creates publisher."""
        self.create_publisher(msg_type,topic,self.qos_profile)
    
    def __add_subscriber(self, msg_type, topic:str, callback):
        """Creates subscriber."""
        self.create_subscription(msg_type,topic,callback,self.qos_profile)
    
    def __construct_msgs(self):
        """Constructs a dictionaries containing pubs and subs msgs."""
        # Publishers
        for k, v in self.pubs_dict.items():
            self.pubs_msgs[k] = v.msg_type()  # v is a publisher.
        # Subscribers
        for k, v in self.subs_dict.items():
            self.subs_msgs[k] = v.msg_type()  # v is a subscriber.

    def __arduino_field_fb_cb(self, msg):
        """Call back for /arduino_field_fb."""
        # Update the message.
        # Note that the dict key should match its corresponding
        #  subscriber topic_name.
        self.subs_msgs["/arduino_field_fb"].x = msg.x
        self.subs_msgs["/arduino_field_fb"].y = msg.y
        self.subs_msgs["/arduino_field_fb"].z = msg.z

    def publish_all(self, *, arduino_field_cmd):
        """Publishes all given messages."""
        # Update message values.
        # Note that dict keys should match their corresponding publisher
        # topic_name.
        self.pubs_msgs["/arduino_field_cmd"].x = arduino_field_cmd[0]
        self.pubs_msgs["/arduino_field_cmd"].y = arduino_field_cmd[1]
        self.pubs_msgs["/arduino_field_cmd"].z = arduino_field_cmd[2]
        # Publish topics.
        self.pubs_dict["/arduino_field_cmd"].publish(
                                    self.pubs_msgs["/arduino_field_cmd"])
    
    def get_subs_values(self):
        """Return the current value of subscribers as a tuple."""
        arduino_field_fb=np.array([self.subs_msgs["/arduino_field_fb"].x,
                                   self.subs_msgs["/arduino_field_fb"].y,
                                   self.subs_msgs["/arduino_field_fb"].z])
        return arduino_field_fb
    
    def __timer_callback(self):
        """This contains the hardware communication loop."""
        # Read last values from pipeline.
        field, states = self.pipeline.get_cmd()
        cmd_mode = self.pipeline.get_cmd_mode()
        # Publish command
        self.publish_all(arduino_field_cmd=field)
        # Read latest subscriber values.
        field_fb = self.get_subs_values()
        # printing
        time_counter_msg = f"{time.time()%1e4:+010.3f}|{self.counter:+010d}|"
        cmd_msg = (",".join(f"{elem:+07.2f}" for elem in field) + "|"
                  +",".join(f"{elem:+07.2f}" for elem in field_fb) + "|"
                  +",".join(f"{elem:+07.2f}" for elem in states[0]) + "|"
                  +f"{states[1]*180/np.pi:+07.2f},{states[2]*180/np.pi:+07.2f}, "
                  +f"{states[3]:01d}")
        print(time_counter_msg,end="")
        print(cmd_msg)
        self.counter = (self.counter + 1)%1000000

class MainExecutor(rclpy.executors.MultiThreadedExecutor):
    """Main executor for arduino comunications."""
    def __init__(self, rate = 100):
        # If rclpy.init() is not called, call it.
        if rclpy.ok() is not True:
            rclpy.init(args = sys.argv)
        super().__init__()
        #
        pipeline = Pipeline(3)
        # Add nodes.
        self.add_node(Peripherals(pipeline, rate = rate))
        
    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True:
            rclpy.shutdown()
        # Catch exceptions
        if exc_type == KeyboardInterrupt:
            print("Interrupted by user.")
            return True

########## Test section ################################################
if __name__ == "__main__":
    with MainExecutor(50) as executor:
        #executor = MainExecutor(50)
        executor.spin()
