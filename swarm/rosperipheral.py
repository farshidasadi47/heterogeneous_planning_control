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

from std_msgs.msg import Int32
from geometry_msgs.msg import Point32
########## Definiitons #################################################
class Peripherals(Node):
    """Main executor for arduino comunications."""
    def __init__(self, rate = 100, name="peripherals"):
        # If rclpy.init() is not called, call it.
        if rclpy.ok() is not True:
            rclpy.init(args = sys.argv)
        super().__init__(name)
        # Some global variable
        self.counter = 0
        # QoS profile
        self.qos_profile = QoSProfile(
        reliability=
                   QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        history = QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        depth=1)
        #self.qos_profile = rclpy.qos.qos_profile_system_default.reliability
        # Add arduino publishers.
        self.__add_publisher(Int32,"/arduino_cmd")
        self.__add_publisher(Point32,"/arduino_field_cmd")
        # Order of adding subscriber and times matters.
        # Add arduino subscribers.
        self.__add_subscriber(Int32,"/arduino_fb", self.__arduino_fb_cb)
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
        self.create_publisher(msg_type,topic,1)#self.qos_profile)
    
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

    def __arduino_fb_cb(self, msg):
        """Call back for /arduino_fb."""
        # Update the message.
        # Note that the name of the message should match its
        # corresponding subscriber topic name.
        self.subs_msgs["/arduino_fb"].data = msg.data

    def __arduino_field_fb_cb(self, msg):
        """Call back for /arduino_field_fb."""
        # Update the message.
        # Note that the dict key should match its corresponding
        #  subscriber topic_name.
        self.subs_msgs["/arduino_field_fb"].x = msg.x
        self.subs_msgs["/arduino_field_fb"].y = msg.y
        self.subs_msgs["/arduino_field_fb"].z = msg.z

    def publish_all(self, *, arduino_cmd: int,
                             arduino_field_cmd):
        """Publishes all given messages."""
        # Update message values.
        # Not that dict keys should match their corresponding publisher
        # topic_name.
        self.pubs_msgs["/arduino_cmd"].data = arduino_cmd
        self.pubs_msgs["/arduino_field_cmd"].x = arduino_field_cmd[0]
        self.pubs_msgs["/arduino_field_cmd"].y = arduino_field_cmd[1]
        self.pubs_msgs["/arduino_field_cmd"].z = arduino_field_cmd[2]
        # Publish topics.
        self.pubs_dict["/arduino_cmd"].publish(
                                          self.pubs_msgs["/arduino_cmd"])
        self.pubs_dict["/arduino_field_cmd"].publish(
                                    self.pubs_msgs["/arduino_field_cmd"])
    
    def get_subs_values(self):
        """Return the current value of subscribers as a tuple."""
        arduino_fb = self.subs_msgs["/arduino_fb"].data
        arduino_field_fb=np.array([self.subs_msgs["/arduino_field_fb"].x,
                                   self.subs_msgs["/arduino_field_fb"].y,
                                   self.subs_msgs["/arduino_field_fb"].z])
        return (arduino_fb, arduino_field_fb)
    
    def __timer_callback(self):
        """This contains the main control loop."""
        sent = np.around(np.random.uniform(-90,90,3),2)
        sent[2] = 100.0
        self.publish_all(arduino_cmd=self.counter, arduino_field_cmd=sent)
        counter_fb, field_fb = self.get_subs_values()
        print_str = (f"{time.time()%1e4:+010.3f}|{self.counter:+010d}, {counter_fb:+010d}, "
                    +",".join(f"{element:+06.2f}" for element in sent)+", "
                    +",".join(f"{element:+06.2f}" for element in field_fb))
        print(print_str)
        self.counter = (self.counter + 1)%1000000

class MainExecutor(rclpy.executors.SingleThreadedExecutor):
    """Main executor for arduino comunications."""
    def __init__(self, rate = 100):
        # If rclpy.init() is not called, call it.
        if rclpy.ok() is not True:
            rclpy.init(args = sys.argv)
        super().__init__()
        # First create the node and its related functionalities.
        self.node = Peripherals(rate = rate)
        # Add nodes.
        self.add_node(self.node)
        
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
# %%
