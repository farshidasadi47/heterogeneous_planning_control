#%%
########################################################################
# This module is responsible for classes and methods that publish
# and subscribe to the peripherals (arduino and probably camera).
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import sys
import time
import re

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Point32
from std_srvs.srv import Empty

# from "foldername" import filename, this is for ROS compatibility.
from swarm import controller, model
########## Definiitons #################################################
class Peripherals(Node):
    """Main executor for arduino comunications."""
    def __init__(self, pipeline: controller.Pipeline,
                       rate = 100, name="peripherals"):
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
        self._add_publisher(Point32,"/arduino_field_cmd")
        # Order of adding subscriber and times matters.
        # Add arduino subscribers.
        self._add_subscriber(Point32,"/arduino_field_fb",
                                                    self._arduino_field_fb_cb)
        # Add timer
        self.timer = self.create_timer(1/rate, self._timer_callback)
        # Get all publishers and subscribers.
        self.pubs_dict = {pub.topic_name:pub for pub in self.publishers}
        self.pubs_dict.pop('/parameter_events')
        self.subs_dict = {sub.topic_name:sub for sub in self.subscriptions}
        # Cusdtruct publisher and subscription instance variables.
        self.pubs_msgs = dict()
        self.subs_msgs = dict()
        self._construct_msgs()

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
    
    def _add_publisher(self, msg_type, topic: str):
        """Creates publisher."""
        self.create_publisher(msg_type,topic,self.qos_profile)
    
    def _add_subscriber(self, msg_type, topic:str, callback):
        """Creates subscriber."""
        self.create_subscription(msg_type,topic,callback,self.qos_profile)
    
    def _construct_msgs(self):
        """Constructs a dictionaries containing pubs and subs msgs."""
        # Publishers
        for k, v in self.pubs_dict.items():
            self.pubs_msgs[k] = v.msg_type()  # v is a publisher.
        # Subscribers
        for k, v in self.subs_dict.items():
            self.subs_msgs[k] = v.msg_type()  # v is a subscriber.

    def _arduino_field_fb_cb(self, msg):
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
    
    def _timer_callback(self):
        """This contains the hardware communication loop."""
        # Read last values from pipeline.
        field, states = self.pipeline.get_cmd()
        cmd_mode = self.pipeline.get_cmd_mode()
        # Publish command
        self.publish_all(arduino_field_cmd=field)
        # Read latest subscriber values.
        field_fb = self.get_subs_values()
        # printing
        time_counter_msg = f"{time.time()%1e3:+08.3f}|{self.counter:06d}|"
        cmd_msg = (",".join(f"{elem:+07.2f}" for elem in field) + "|"
                  +",".join(f"{elem:+07.2f}" for elem in field_fb) + "|"
                  +",".join(f"{elem:+07.2f}" for elem in states[0]) + "|"
                  +f"{states[1]*180/np.pi:+07.2f},{states[2]*180/np.pi:+07.2f}, "
                  +f"{states[3]:01d}")
        if cmd_mode == "server":
            print(time_counter_msg,end="")
            print(cmd_msg)
        self.counter = (self.counter + 1)%1000000

class ControlService(Node):
    """This class holds services that control swarm of milirobot."""
    def __init__(self, pipeline: controller.Pipeline,
                       control: controller.Controller, rate = 100):
        # If rclpy.init() is not called, call it.
        if rclpy.ok() is not True:
            rclpy.init(args = sys.argv)
        super().__init__("controlservice")
        self.pipeline = pipeline
        self.control = control
        self.rate = self.create_rate(rate)
        # Action servers
        self._add_service_server(Empty,'set_idle', self._set_idle_server_cb)
        self._add_service_server(Empty,'/feedfrwd_input',
                                               self._feedfrwd_input_server_cb)
        self._add_service_server(Empty,'/feedfrwd_single',
                                              self._feedfrwd_single_server_cb)
        self._add_service_server(Empty,'/pivot_walking',
                                                self._pivot_walking_server_cb)
        self._add_service_server(Empty,'/mode_change',
                                                self._mode_change_server_fb)
        self._add_service_server(Empty,'/tumbling',
                                                self._tumbling_server_fb)
        self._add_service_server(Empty,'/set_steps',
                                               self._set_steps_server_cb)
    
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

    def _add_service_server(self, srv_type, srv_name: str, callback):
        """Adds an action server and puts it into a dictionary."""
        self.create_service(srv_type, srv_name, callback)

    # Callbacks
    # Servers
    def _set_idle_server_cb(self, request, response):
        """
        Sets field command in idle condition.
        """
        print("*"*72)
        regex = r'([+-]?\d+\.?\d*(, *| +)){2}([+-]?\d+\.?\d* *)'
        field = np.array([0.0,0,0.0])
        self.rate.sleep()
        while True:
            try:
                print("Enter field angles and percentage: theta, alpha, %power.")
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
                field = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                str_msg = (f"[theta, alpha, %power] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in field) + "]")
                print(str_msg)
            except:
                print("Ooops! values ignored. Enter values like the template.")
            self.pipeline.set_idle(field)
        print("*"*72)
        return response

    def _feedfrwd_input_server_cb(self, request, response):
        """
        This service calls feedforward_line function of Controller
         class and executes a given input_series.
        """
        input_series = np.array([[10,0,1],
                                 [10,np.pi/2,-2],
                                 [10,0,2],
                                 [10,0,0]])
        # Check if we are not in the middle of another service that 
        # calls data pipeline, If we are, ignore this service call.
        interactive = True
        if self.pipeline.get_cmd_mode() == "idle":
            try:
                # Compatibility check,raises ValueError if incompatible.
                self.control.line_input_compatibility_check(input_series,
                                                            interactive)
                # Change command mode.
                self.pipeline.set_cmd_mode("server")
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field, states in self.control.feedforward_line(
                                                                 input_series,
                                                                 interactive):
                    self.pipeline.set_cmd(np.array([*field, 100.0]))
                    self.pipeline.set_state(states)
                    self.rate.sleep()
                # set command to zero
                self.pipeline.set_cmd(np.zeros(3,dtype=float))
                self.pipeline.set_state(states)
                self.rate.sleep()
            except ValueError:
                # ValueError can be raised by input_compatibility_check.
                # This error is handled internally, so we pass here.
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Release the data pipeline.
        self.pipeline.set_cmd_mode("idle")
        self.rate.sleep()
        return response

    def _feedfrwd_single_server_cb(self, request, response):
        """
        This service calls feedforward_line function of Controller
         class and executes a given input_series in single_step mode.
        """
        input_series = np.array([[50,0,1],
                                 [10,np.pi/2,-2],
                                 [10*4,np.pi,0],
                                 [50,-np.pi/2,2],
                                 [10,-np.pi/2,-1]])
        regex = r'^q'
        msg_str = "Enter \"q\" for quitting, anything else for proceeding: "
        num_sections = input_series.shape[0]
        last_section = False
        # Check if we are not in the middle of another service that 
        # calls data pipeline, If we are, ignore this service call.
        interactive = True
        if self.pipeline.get_cmd_mode() == "idle":
            try:
                # Compatibility check,raises ValueError if incompatible.
                self.control.line_input_compatibility_check(input_series,
                                                            interactive)
                for section in range(num_sections):
                    if section == (num_sections - 1):
                        # If last section,call with robot final line up.
                        last_section = True
                    # Print current section.
                    current_input = input_series[[section],:]
                    input_str = ",".join(f"{elem:+07.2f}"
                                                for elem in current_input[0,:])
                    print('Current section input: ' + input_str)
                    # Get the input.
                    in_str = input(msg_str).strip()
                    # Check if user requests quitting.
                    if re.fullmatch(regex,in_str) is not None:
                        print("Exitted \"service server\".")
                        break
                    # Change command mode.
                    self.pipeline.set_cmd_mode("server")
                    self.rate.sleep()
                    # Execute main controller and update pipeline.
                    for field, states in self.control.feedforward_line(
                                       current_input,interactive,last_section):
                        self.pipeline.set_cmd(np.array([*field,100.0]))
                        self.pipeline.set_state(states)
                        self.rate.sleep()
                    # set command to zero
                    self.pipeline.set_cmd(np.zeros(3,dtype=float))
                    self.pipeline.set_state(states)
                    # Release the data pipeline.
                    self.pipeline.set_cmd_mode("idle")
                    self.rate.sleep()
            except Exception as exc:
                print("Ooops! exception happened. Values are ignored.")
                print("Exception details:")
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Release the data pipeline.
        self.pipeline.set_cmd_mode("idle")
        self.rate.sleep()
        return response

    def _pivot_walking_server_cb(self, request, response):
        """
        This service calls pivot_walking function.
        """
        # Check if we are not in the middle of another service that 
        # calls data pipeline, If we are, ignore this service call.
        alternative = True
        print("*"*72)
        regex = r'([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        if self.pipeline.get_cmd_mode() == "idle":
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
                params = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                str_msg = (f"[n_steps, starting_theta] = ["
                         + ",".join(f"{elem:+07.2f}" for elem in params) + "]")
                print(str_msg)
                # Change command mode.
                self.pipeline.set_cmd_mode("server")
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field, states in self.control.pivot_walking_walk(params,
                                                                  alternative):
                    self.pipeline.set_cmd(np.array([*field, 100.0]))
                    self.pipeline.set_state(states)
                    self.rate.sleep()
                # set command to zero
                self.pipeline.set_cmd(np.zeros(3,dtype=float))
                self.pipeline.set_state(states)
                self.rate.sleep()
            except ValueError:
                # This error is handled internally, so we pass here.
                pass
            except AssertionError:
                # Handled internally and no further action is needed.
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        # Release the data pipeline and neutral magnetic field.
        self.pipeline.set_idle(np.zeros(3,dtype=float))
        self.pipeline.set_cmd_mode("idle")
        self.rate.sleep()
        return response

    def _mode_change_server_fb(self, request, response):
        """
        This service calls mode_changing function and performs one
        mode change.
        """
        print("*"*72)
        regex = r'([+-]?\d+(, *| +)){1}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        # Check if we are not in the middle of another service that 
        # calls data pipeline, If we are, ignore this service call.
        if self.pipeline.get_cmd_mode() == "idle":
            try:
                # Get the input.
                print("Enter params: starting_mode, starting_theta.")
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
                # Parse user input and reset mode, theta.
                params = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                str_msg = (f"[starting_mode, starting_theta] = ["
                         + ",".join(f"{elem:+07.2f}" for elem in params) + "]")
                print(str_msg)
                self.control.reset_state(theta=np.deg2rad(params[1]),
                                         mode=int(params[0]))
                # Get next mode.
                des_mode = self.control.mode_sequence[1]
                # Change command mode.
                self.pipeline.set_cmd_mode("server")
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field, states in self.control.mode_changing(des_mode,
                                                                       0,False):
                    self.pipeline.set_cmd(np.array([*field, 100.0]))
                    self.pipeline.set_state(states)
                    self.rate.sleep()
                # set command to zero
                self.pipeline.set_cmd(np.zeros(3,dtype=float))
                self.pipeline.set_state(states)
                self.rate.sleep()
            except ValueError:
                # This error is handled internally, so we pass here.
                pass
            except AssertionError:
                # Handled internally and no further action is needed.
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        # Release the data pipeline and neutral magnetic field.
        self.pipeline.set_idle(np.zeros(3,dtype=float))
        self.pipeline.set_cmd_mode("idle")
        self.rate.sleep()
        return response

    def _tumbling_server_fb(self, request, response):
        """
        This service calls tumbling function and performs tumbling.
        """
        print("*"*72)
        regex = r'([+-]?\d+(, *| +)){2}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        # Check if we are not in the middle of another service that 
        # calls data pipeline, If we are, ignore this service call.
        if self.pipeline.get_cmd_mode() == "idle":
            try:
                # Get the input.
                print("Enter params: n_tumble, starting_mode, starting_theta.")
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
                # Parse user input and reset mode, theta.
                params = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                str_msg = (f"[n_tumble, starting_mode, starting_theta] = ["
                         + ",".join(f"{elem:+07.2f}" for elem in params) + "]")
                print(str_msg)
                self.control.reset_state(theta=np.deg2rad(params[2]),
                                         mode=int(params[1]))
                # Process command
                des_r = params[0]*self.control.specs.tumbling_length
                input_cmd = np.array([des_r, 0.0, 0])
                # Change command mode.
                self.pipeline.set_cmd_mode("server")
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field, states in self.control.tumbling(input_cmd,
                                                                 False, False):
                    self.pipeline.set_cmd(np.array([*field, 100.0]))
                    self.pipeline.set_state(states)
                    self.rate.sleep()
                # set command to zero
                self.pipeline.set_cmd(np.zeros(3,dtype=float))
                self.pipeline.set_state(states)
                self.rate.sleep()
            except ValueError:
                # This error is handled internally, so we pass here.
                pass
            except Exception as exc:
                print("Ooops! exception happened. Values are ignored.")
                print("Exception details:")
                print(type(exc).__name__,exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        # Release the data pipeline and neutral magnetic field.
        self.pipeline.set_idle(np.zeros(3,dtype=float))
        self.pipeline.set_cmd_mode("idle")
        self.rate.sleep()
        return response
    
    def _set_steps_server_cb(self, request, response):
        """
        Sets steps parameters of the Controller class.
        """
        print("*"*72)
        print("Current values of parameters:\n"+
            f"theta_inc = {np.rad2deg(self.control.theta_step_inc)}\n"+
            f"theta_rot_inc = {np.rad2deg(self.control.theta_rot_step_inc)}\n"+
            f"alpha_inc = {np.rad2deg(self.control.alpha_step_inc)}\n"+
            f"pivot_inc = {np.rad2deg(self.control.pivot_step_inc)}\n"+
            f"sweep_theta = {np.rad2deg(self.control.sweep_theta)}\n"+
            f"sweep_alpha = {np.rad2deg(self.control.sweep_alpha)}\n"+
            f"tumble_inc = {np.rad2deg(self.control.tumble_step_inc)}")
        regex = r'([+-]?\d+\.?\d*(, *| +)){6}([+-]?\d+\.?\d* *)'
        self.rate.sleep()
        if self.pipeline.get_cmd_mode() == "idle":
            try:
                print("Enter new values, separated by comma or space.")
                print("Enter \"q\" for quitting.")
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match('q',in_str) is not None:
                    print("Exitted \"service server\".")
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex,in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse input and change parameter.
                params = list(map(float,re.findall(r'[+-]?\d+\.?\d*',in_str)))
                self.control.theta_step_inc = np.deg2rad(params[0])
                self.control.theta_rot_step_inc = np.deg2rad(params[1])
                self.control.alpha_step_inc = np.deg2rad(params[2])
                self.control.pivot_step_inc = np.deg2rad(params[3])
                self.control.sweep_theta = np.deg2rad(params[4])
                self.control.sweep_alpha = np.deg2rad(params[5])
                self.control.tumble_step_inc = np.deg2rad(params[6])
                print("Parameters changed to:\n"+
            f"theta_inc = {np.rad2deg(self.control.theta_step_inc)}\n"+
            f"theta_rot_inc = {np.rad2deg(self.control.theta_rot_step_inc)}\n"+
            f"alpha_inc = {np.rad2deg(self.control.alpha_step_inc)}\n"+
            f"pivot_inc = {np.rad2deg(self.control.pivot_step_inc)}\n"+
            f"sweep_theta = {np.rad2deg(self.control.sweep_theta)}\n"+
            f"sweep_alpha = {np.rad2deg(self.control.sweep_alpha)}\n"+
            f"tumble_inc = {np.rad2deg(self.control.tumble_step_inc)}")
                self.rate.sleep()
            except ValueError:
                pass
            except Exception as exc:
                print("Ooops! exception happened. Values are ignored.")
                print("Exception details:")
                print(type(exc).__name__,exc.args)
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*"*72)
        return response

class MainExecutor(rclpy.executors.MultiThreadedExecutor):
    """Main executor for arduino comunications."""
    def __init__(self, rate = 100):
        # If rclpy.init() is not called, call it.
        if rclpy.ok() is not True:
            rclpy.init(args = sys.argv)
        super().__init__()
        #
        specs = model.SwarmSpecs.robo3()
        pipeline = controller.Pipeline()
        control = controller.Controller(specs,np.array([0,0,20,0,40,0]),0,1)
        # Set initialize pipeline states.
        pipeline.set_state(control.get_state())
        # Add nodes.
        self.add_node(Peripherals(pipeline, rate = rate))
        self.add_node(ControlService(pipeline, control, rate))
        
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
