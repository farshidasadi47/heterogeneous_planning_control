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

class ControlNode(NodeTemplate):
    """Contains all control related service servers."""
    def __init__(self, control: closedloop.Controller, rate = 50):
        super().__init__("controlservice")
        self.control = control
        self.cmd_mode = "idle"
        self.rate = self.create_rate(rate)
        # Add publishers and subscribers.
        self._add_publisher(Point32,"/arduino_field_cmd")
        self._add_subscriber(Point32,"/arduino_field_fb",
                                                    self._arduino_field_fb_cb)
        self._add_subscriber(Float32MultiArray,"/state_fb",
                                            self._state_fb_cb, reliable= False)
        # Custruct publisher and subscription instance variables.
        self._construct_pubsub_msgs()
        # Service and action servers
        self._add_service_server(Empty,'set_idle', self._set_idle_server_cb)
        # Open loops with information from camera.
        self._add_action_server(RotateAbsolute,'/pivot_walking',
                                                self._pivot_walking_server_cb)
        self._add_action_server(RotateAbsolute, '/mode_change',
                                                self._mode_change_server_fb)
        self._add_action_server(RotateAbsolute,'/tumbling',
                                                self._tumbling_server_fb)
        # Closed loops
        self._add_action_server(RotateAbsolute,'cart_pivot',
                                                  self._cart_pivot_server_cb)
        self._add_action_server(RotateAbsolute,'calibration_pivot',
                                             self._calibration_pivot_server_cb)
    
    def _arduino_field_fb_cb(self, msg):
        """Call back for /arduino_field_fb."""
        # Reads arduino field feedback.
        self.subs_msgs["/arduino_field_fb"].x = msg.x
        self.subs_msgs["/arduino_field_fb"].y = msg.y
        self.subs_msgs["/arduino_field_fb"].z = msg.z
    
    def _state_fb_cb(self,msg):
        self.subs_msgs['/state_fb'].data = msg.data

    def publish_field(self, field):
        """Publishes all given messages."""
        # Update all values to be published. 
        self.pubs_msgs["/arduino_field_cmd"].x = field[0]
        self.pubs_msgs["/arduino_field_cmd"].y = field[1]
        self.pubs_msgs["/arduino_field_cmd"].z = field[2]
        # Publish topics.
        self.pubs_dict["/arduino_field_cmd"].publish(
                                    self.pubs_msgs["/arduino_field_cmd"])
    
    def get_subs_values(self):
        """Return the current value of subscribers as an array."""
        field_fb= [self.subs_msgs["/arduino_field_fb"].x,
                   self.subs_msgs["/arduino_field_fb"].y,
                   self.subs_msgs["/arduino_field_fb"].z]
        feedback = self.subs_msgs['/state_fb'].data 
        return field_fb, feedback
    
    def print_stats(self, field, field_fb, state, state_fb, cnt):
        msg = (
             f"{time.time()%1e3:+8.3f}|{cnt:05d}|"
            +"".join(f"{elem:+07.2f}" for elem in field[:2]) + "|"
            +"".join(f"{elem:+07.2f}" for elem in field_fb[:2]) + "|"
            +"".join(f"{elem:+07.2f}" for elem in state_fb[0]) + "|"
            + f"{np.rad2deg(state_fb[1]):+07.2f}" + '|'
            +"".join(f"{elem:+07.2f}" for elem in state[0]) + "|"
            +f"{np.rad2deg(state[1]):+07.2f},{np.rad2deg(state[2]):+07.2f}| "
            +f"{state[3]:01d}"
            )
        print(msg)

    # Service and action servar callbacks
    def _set_idle_server_cb(self, request, response):
        """
        Sets field command in idle condition.
        """
        print("*"*72)
        regex = r'([+-]?\d+\.?\d*(, *| +)){2}([+-]?\d+\.?\d* *)'
        field = [0.0,0.0,0.0]
        self.cmd_mode = "idle"
        self.rate.sleep()
        while True:
            try:
                print("Enter body angles deg and power: theta, alpha, %power.")
                print("Enter \"q\" for quitting.")
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    print("Exitted \"set_idle\".")
                    break
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    raise ValueError
                # Parse user input.
                body =list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                field= self.control.body2magnet(np.deg2rad(body[:2]))+[body[2]]
                msg1 = (f"body[theta, alpha, %power] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in body) + "], ")
                msg2 = (f"field[theta, alpha, %power] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in field) + "]")
                print(msg1 + msg2)
            except:
                print("Ooops! values ignored. Enter values like the template.")
            self.publish_field(field)
        self.publish_field([0.0]*3)
        print("*"*72)
        return response
    
    def _pivot_walking_server_cb(self, goal_handle):
        """
        This service calls pivot_walking function.
        """
        # Ignore the call if we are in the middle of another process.
        cnt = 0
        field = [0.0]*3
        print("*"*72)
        regex = r'([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode = "busy"
            try:
                # Get the input.
                print("Enter params: num_steps, starting_theta (in degrees)")
                print("Enter \"q\" for quitting.")
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    print("Exitted \"service server\".")
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                i_cmd = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                msg = (f"[n_steps, starting_theta] = ["
                         + ",".join(f"{elem:+07.2f}" for elem in i_cmd) + "]")
                print(msg)
                # Get current position
                _, feedback = self.get_subs_values()
                self.publish_field(field)
                xi, theta_i = self.control.process_robots(feedback)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                self.rate.sleep()
                # Change command mode and execute.
                i_cmd[1] = np.deg2rad(i_cmd[1])
                iterator=self.control.pivot_walking_walk(i_cmd,line_up=False)
                self.rate.sleep()
                # Execute main controllr.
                for field in iterator:
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # Get the first robot found.
                    field_fb, feedback = self.get_subs_values()
                    state_fb = self.control.process_robots(feedback)
                    self.publish_field(field)
                    self.print_stats(field, field_fb, state, state_fb,cnt)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb,cnt)
                self.rate.sleep()
                print(msg_i)
                _ , feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                polar_dist = self.control.get_polar_cmd(xi,xf,average=False)
                polar_dist[1] = np.rad2deg(polar_dist[1])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                # Set commands to zero.
                self.publish_field([0.0]*3)
                self.rate.sleep()
            except Exception as exc:
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0]*3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _mode_change_server_fb(self, goal_handle):
        """
        This service calls mode_changing function and performs one
        mode change.
        """
        print("*"*72)
        cnt = 0
        field = [0.0]*3
        regex = r'([+-]?\d+(, *| +)){1}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        # Ignore the call if we are in the middle of another process.
        if self.cmd_mode == "idle":
            self.cmd_mode = "busy"
            try:
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                xi, theta_i = state_fb
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                self.control.reset_state(pos=xi, theta = theta_i)
                self.rate.sleep()
                # Get the input.
                print("Enter params: phi (degrees), mode.")
                print("Enter \"q\" for quitting.")
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    raise ValueError("Exitted \"service server\".")
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    raise ValueError("Invalid input. Exitted service request.")
                # Parse user input and reset mode, theta.
                params = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                str_msg = (f"[phi, mode] = ["
                         + ",".join(f"{elem:+07.2f}" for elem in params) + "]")
                print(str_msg)
                self.publish_field(field)
                self.rate.sleep()
                # Change command mode and execute.
                iterator= self.control.mode_changing([0.0,
                                              np.deg2rad(params[0]),params[1]])
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field in iterator:
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # Get the first robot found.
                    field_fb, feedback = self.get_subs_values()
                    #state_fb = self.control.process_robots(feedback)
                    self.publish_field(field)
                    self.print_stats(field, field_fb, state, state_fb,cnt)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb,cnt)
                self.rate.sleep()
                print(msg_i)
                _ , feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                polar_dist = self.control.get_polar_cmd(xi,xf,average=False)
                polar_dist[1] = np.rad2deg(polar_dist[1])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                # Set commands to zero.
                self.publish_field([0.0]*3)
                self.rate.sleep()
            except Exception as exc:
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0]*3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result
    
    def _tumbling_server_fb(self, goal_handle):
        """
        This service calls tumbling function and performs tumbling.
        """
        print("*"*72)
        cnt = 0
        field = [0.0]*3
        regex = r'([+-]?\d+(, *| +)){1}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        # Check if we are not in the middle of another service that 
        # calls data pipeline, If we are, ignore this service call.
        if self.cmd_mode == "idle":
            self.cmd_mode = "busy"
            try:
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                xi, theta_i = state_fb
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                self.control.reset_state(pos=xi, theta = theta_i)
                self.rate.sleep()
                # Get the input.
                print("Enter params: n_tumble, phi.")
                print("Enter \"q\" for quitting.")
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    raise ValueError("Exitted \"service server\".")
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    raise ValueError("Invalid input. Exitted service request.")
                # Parse user input and reset mode, theta.
                params = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                str_msg = (f"[n_tumble, phi] = ["
                         + ",".join(f"{elem:+07.2f}" for elem in params) + "]")
                print(str_msg)
                params[0] = params[0]*self.control.specs.tumbling_length
                params[1] = np.deg2rad(params[1])
                self.publish_field(field)
                self.rate.sleep()
                # Get controller.
                iterator = self.control.tumbling(params)
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field in iterator:
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # Get the first robot found.
                    field_fb, feedback = self.get_subs_values()
                    if abs(state[2]) <0.01:
                        state_fb = self.control.process_robots(feedback)
                    self.publish_field(field)
                    self.print_stats(field, field_fb, state, state_fb,cnt)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb,cnt)
                self.rate.sleep()
                print(msg_i)
                _ , feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                polar_dist = self.control.get_polar_cmd(xi,xf,average=False)
                polar_dist[1] = np.rad2deg(polar_dist[1])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                # Set commands to zero.
                self.publish_field([0.0]*3)
                self.rate.sleep()
            except Exception as exc:
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0]*3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _cart_pivot_server_cb(self,goal_handle):
        """Executes closed loop pivot alking for one robot."""
        cnt = 0
        print("*"*72)
        regex = r'([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)'
        field = [0.0,0.0,0.0]
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                print("Enter final position: [xf, yf].")
                print("Enter \"q\" for quitting.")
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    print("Exitted \"set_idle\".")
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                xf =list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                xf *= self.control.specs.n_robot
                # Get current position
                _, feedback = self.get_subs_values()
                xi, theta_i = self.control.process_robots(feedback)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                msg_f = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                print(msg_f)
                # Reset state
                self.control.reset_state(pos = xi, theta = theta_i)
                self.publish_field(field)
                self.rate.sleep()
                # Execute closed loop control.
                iterator = self.control.closed_pivot_cart(xf)
                self.rate.sleep()
                for field in iterator:
                    # Get current position
                    field_fb, feedback = self.get_subs_values()
                    state_fb = self.control.process_robots(feedback)
                    if field is None:
                        field = iterator.send(state_fb)
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    self.print_stats(field, field_fb, state, state_fb,cnt)
                    self.publish_field(field)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb,cnt)
                self.rate.sleep()
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg_i)
                print(msg_f)
                print(msg)
                polar_dist = self.control.get_polar_cmd(xi,xf)
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                self.publish_field([0.0]*3)
                self.rate.sleep()
            except StopIteration: pass
            except Exception as exc:
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0]*3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _calibration_pivot_server_cb(self,goal_handle):
        cnt = 0
        print("*"*72)
        regex = r'([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)'
        field = [0.0,0.0,0.0]
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                print("Enter final position: [r, n_sections].")
                print("Enter \"q\" for quitting.")
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    print("Exitted \"set_idle\".")
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                params =list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                # Get current position
                _, feedback = self.get_subs_values()
                xi, theta_i = self.control.process_robots(feedback)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                print(f"r: {params[0]:+07.2f}, n_sections:{params[1]:+07.2f}")
                # Reset state
                self.control.reset_state(pos = xi, theta = theta_i)
                self.publish_field(field)
                self.rate.sleep()
                # Execute closed loop control.
                iterator = self.control.pivot_calibration(params[1])
                self.rate.sleep()
                #for field in iterator:
                while True:
                    field = next(iterator)
                    # Get current position
                    field_fb, feedback = self.get_subs_values()
                    state_fb = self.control.process_robots(feedback)
                    if field is None:
                        field = iterator.send(state_fb)
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    self.print_stats(field, field_fb, state, state_fb,cnt)
                    self.publish_field(field)
                    cnt += 1
                    self.rate.sleep()
            except StopIteration as e:
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb,cnt)
                self.rate.sleep()
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                self.publish_field([0.0]*3)
                self.rate.sleep()
                print(e.value)
                pass
            except Exception as exc:
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0]*3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

class MainExecutor(MultiThreadedExecutorTemplate):
    """Main executor for arduino comunications."""
    def __init__(self, rate = 50):
        super().__init__()
        specs = model.SwarmSpecs.robo3()
        control = closedloop.Controller(specs,np.array([0,0,20,0,40,0]),0,1)
        # Add nodes.
        self.add_node(ControlNode(control, rate))
        print("*"*72 + "\nMain executor is initialized.\n" + "*"*72)

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

def main():
    with MainExecutor(50) as executor:
        executor.spin()

def get_video():
    with GetVideoExecutor(55) as executor:
        executor.spin()

def show_video():
    with ShowVideoExecutor(30) as video:
        video.spin()
########## Test section ################################################
if __name__ == "__main__":
    get_video()
