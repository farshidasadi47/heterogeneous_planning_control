#%%
########################################################################
# This files hold classes and functions that controls swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from dataclasses import dataclass
from itertools import combinations
from collections import deque
from math import remainder
import threading
import time

import numpy as np
import numpy.matlib
import casadi as ca

from scipy.spatial.transform import Rotation

# from "foldername" import filename, this is for ROS compatibility.
from swarm import model

np.set_printoptions(precision=2, suppress=True)
########## classes and functions #######################################
class Controller():
    """
    This class holds controlers for pivot walking and rolling of
    swarm of millirobots.
    """
    # Inheriting from model.Swarm is done, so that it can be used later
    # in planner class in control process. Other-wise it is not a 
    # a crucial part of the architecture.
    def __init__(self, specs: model.SwarmSpecs,
                       pos: np.ndarray, theta: float, mode: int):
        self.specs = specs
        self.__set_rotation_constants_and_functions()
        self.reset_state(pos, theta, 0, mode)
        self.theta_step_inc = np.deg2rad(5)
        self.theta_rot_step_inc = np.deg2rad(5)
        self.alpha_step_inc = np.deg2rad(5)
        self.pivot_step_inc = np.deg2rad(5)
        self.sweep_theta = np.deg2rad(30)  # Max sweep angle
        self.sweep_alpha = np.deg2rad(35)  # alpha sweep limit.
        self.tumble_step_inc = np.deg2rad(5)

    def reset_state(self, pos: np.ndarray = None, theta: float = None,
                          alpha: float = None, mode: int = None):
        if pos is None:
            pos = self.pos
        if theta is None:
            theta = self.theta
        if alpha is None:
            alpha = self.alpha
        if mode is None:
            mode = self.mode
        if (pos.shape[0]//2 != self.specs.n_robot):
            error_message = """Position does not match number of the robots."""
            raise ValueError(error_message)
        #
        pivot_length = self.specs.pivot_length[mode,:]
        leg_vect = np.array([0,self.specs.pivot_length[mode,0]/2,0])
        self.pos = pos.astype(float)
        # Calculate leg positions.
        self.posa = np.zeros_like(self.pos)
        self.posb = np.zeros_like(self.pos)
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        for robot in range(self.specs.n_robot):
            # a is half way along +y or leg_vect.
            self.posa[2*robot:2*robot+2] = (pivot_length[robot]*leg_vect/2
                                            + self.pos[2*robot:2*robot+2])
            # b is half way along -y or -leg_vect
            self.posb[2*robot:2*robot+2] = (-pivot_length[robot]*leg_vect/2
                                            + self.pos[2*robot:2*robot+2])
        self.theta = theta
        self.alpha = alpha
        self.mode = mode
        self.update_mode_sequence(mode)

    def get_state(self):
        return self.pos, self.theta, self.alpha, self.mode
    
    def update_mode_sequence(self,mode: int):
        self.mode_sequence = deque(range(1,self.specs.n_mode))
        self.mode_sequence.rotate(-mode+1)

    def __set_rotation_constants_and_functions(self):
        """
        This function initializes magnets vectors for each mode.
        It also constructs lambda functions for rotating 
        given vectors for given amounnts around certain global axes.
        """
        n_mode = self.specs.n_mode        
        self.rot_vect = np.array([0,1,0])  # Rotation axis when teta=0.
        # Set up rotation functions.
        self.rotx = lambda vect, ang: Rotation.from_euler('x', ang).apply(vect)
        self.roty = lambda vect, ang: Rotation.from_euler('y', ang).apply(vect)
        self.rotz = lambda vect, ang: Rotation.from_euler('z', ang).apply(vect)
        # Rotation around a given axis, angles based on right hand rule.
        self.rotv = (lambda vect, axis, ang: 
                          Rotation.from_rotvec(ang*axis).apply(vect).squeeze())
        # Set magnet vector
        self.magnet_vect = np.array([0.0,-1.0,0.0])

    def set_steps(self, inc_theta, inc_alpha, inc_pivot,
                        sweep_theta, sweep_alpha, inc_tumble):
        """
        Sets theta, alpha, and rolling steps parameters.
        All parameters should be given in Degrees.
        For definition of the angles see the paper.
        """
        self.theta_step_inc = np.deg2rad(inc_theta)
        self.alpha_step_inc = np.deg2rad(inc_alpha)
        self.pivot_step_inc = np.deg2rad(inc_pivot)
        self.tumble_step_inc = np.deg2rad(inc_tumble)
        self.sweep_theta = np.deg2rad(sweep_theta)  # Max sweep angle
        self.sweep_alpha = np.deg2rad(sweep_alpha)  # alpha sweep limit.
    
    def angle_body_to_magnet(self, ang: np.ndarray):
        """
        Converts (theta, alpha) of robots body, to (theta, alpha) of 
        the robots magnets. The converted value can be used as desired
        orientation of coils magnetic field.
        @param: array composed of theta, and alpha of robot body in
                Radians.
         @type: 1D numpy array.
        """
        # Get magnet vector.
        magnet_vect = self.magnet_vect
        # Calculate the cartesian magnet vetor.
        # Rotate alpha around X axis.
        magnet_vect = self.rotx(magnet_vect, ang[1])
        # Rotate theta around Z axis.
        magnet_vect = self.rotz(magnet_vect, ang[0])
        # Convert to spherical coordinate, [theta_m, alpha_m].
        magnet_sph = self.cart_to_sph(magnet_vect)
        return magnet_sph
    
    @staticmethod
    def cart_to_sph(cartesian):
        # alpha: arctan(z/(x**2 + y**2)**.5)
        alpha = np.degrees(np.arctan2(cartesian[2],
                                      np.linalg.norm(cartesian[:2])))
        # theta: arctan(y/x)
        theta = np.degrees(np.arctan2(cartesian[1], cartesian[0]))
        return np.array([theta, alpha])
    
    @staticmethod
    def round_to_zero(x: float, precision: int = 16):
        """Rounds down numbers with given precision."""
        return int(x*(10**precision))/(10**precision)
    
    @staticmethod
    def frange(start, stop=None, step=None):
        # This function is copied from web.
        # if set start=0.0 and step = 1.0 if not specified
        start = float(start)
        if stop == None:
            stop = start + 0.0
            start = 0.0
        if step == None:
            step = 1.0
        #
        count = 0
        while True:
            temp = float(start + count * step)
            if step > 0 and temp >= stop:
                break
            elif step < 0 and temp <= stop:
                break
            yield temp
            count += 1

    @staticmethod
    def wrap(angle):
        """Wraps angles between -PI to PI."""
        wrapped = remainder(-angle+np.pi, 2*np.pi)
        if wrapped< 0:
            wrapped += 2*np.pi
        wrapped -= np.pi
        return -wrapped
    
    @staticmethod
    def wrap_range(from_ang, to_ang, inc = 1):
        """
        Yields a range of wrapped angle that goes from \"from_ang\"
        to \"to_ang\".
        """
        diff = float(Controller.wrap(to_ang - from_ang))
        if diff < 0:
            inc *= -1
        diff = Controller.round_to_zero(diff)
        to_ang = from_ang + diff
        for ang in Controller.frange(from_ang, to_ang, inc):
            yield Controller.wrap(ang)
    
    @staticmethod
    def wrap_range_twin(*,from_1, to_1, inc_1, from_2, to_2, inc_2):
        """
        Yields a range of wrapped angle that goes from \"from_i\"
        to \"to_i\" simoultaneously.
        """
        assert_str = "Cannot handle cases that \"from_i == to_i\" is True."
        assert (from_1 != to_1 and from_2!= to_2), assert_str
        diff_1 = float(Controller.wrap(to_1 - from_1))
        diff_2 = float(Controller.wrap(to_2 - from_2))
        # modify sign of increments if necessary.
        if diff_1 < 0:
            inc_1 *= -1
        if diff_2 < 0:
            inc_2 *= -1
        # Round up to a precision to avoid numerical problems.
        diff_1 = Controller.round_to_zero(diff_1)
        diff_2 = Controller.round_to_zero(diff_2)
        #
        to_1 = from_1 + diff_1
        to_2 = from_2 + diff_2
        # Determine number of steps in range.
        n_1 = np.ceil(diff_1/inc_1).astype(int)
        n_2 = np.ceil(diff_2/inc_2).astype(int)
        # Modify increments based on longer range.
        n_longer = max(n_1,n_2)
        if n_longer>0:
            inc_1 = diff_1/n_longer
            inc_2 = diff_2/n_longer
        # Get generators.
        iter_1 = Controller.frange(from_1, to_1, inc_1)
        iter_2 = Controller.frange(from_2, to_2, inc_2)
        # Yield the values.
        for ang_1 in iter_1:
            ang_2 = next(iter_2)
            yield ang_1, ang_2
        # Repeat last, for more delay and accuracy.
        yield ang_1, ang_2
    
    @staticmethod
    def range_oval(sweep_theta, sweep_alpha, inc):
        """
        This function produces a range tuple for theta and alpha.
        The angles are produced so that the lifted end of the robot will
        go on an half oval path. The oval majpr axes lengths are as:
        horizontal ~ tan(sweep_theta)
        vertical   ~ tan(sweep_alpha)
        """
        assert_str = ("\nThis functions is designed to accept following ranges:"
                    +"\n\t0 =< sweep_thata =< pi/3"
                    +"\n\t0 =< sweep_alpha =< pi/3" + "\n\t0 < inc")
        assert (sweep_theta>=0 and sweep_theta<=np.pi/3 and 
                sweep_alpha>=0 and sweep_alpha<=np.pi/3 and inc>0), assert_str
        #
        fr = lambda ang: (np.tan(sweep_alpha)*np.tan(sweep_theta)
                         /np.sqrt((np.tan(sweep_alpha)*np.cos(ang))**2
                                 +(np.tan(sweep_theta)*np.sin(ang))**2))
        #
        iterator = Controller.frange(0,np.pi,inc)
        iterator = np.arange(0,np.pi,inc).flat
        for ang in iterator:
            # Get corresponding radius on the ellipse.
            r = fr(ang)
            # Calculate angles and yield them.
            alpha = np.arctan2(r*np.sin(ang), 1)
            theta = sweep_theta - np.arctan2(r*np.cos(ang), 1)
            yield theta, alpha
        # Yield last one.
        yield sweep_theta*2, 0.0
        # Repeat last, for more delay and accuracy.
        yield sweep_theta*2, 0.0

    # Control related methods
    def step_alpha(self, desired_alpha:float, inc = None):
        """
        Yields body angles that transitions robot to desired alpha.
        """
        if inc is None:
            inc = self.alpha_step_inc
        starting_alpha = self.alpha
        iterator = self.wrap_range(starting_alpha, desired_alpha, inc)
        for alpha in iterator:
            self.update_alpha(alpha)
            yield np.array([self.theta, alpha])
        # Yield last section.
        alpha = self.wrap(desired_alpha)
        self.update_alpha(alpha)
        yield np.array([self.theta, alpha])
        # Repeat last, for more delay and accuracy.
        if abs(alpha) < 1:
            yield np.array([self.theta, alpha])

    def update_alpha(self, alpha: float):
        self.alpha = Controller.wrap(alpha)

    def step_theta(self, desired_theta:float, pivot: str = None):
        """
        Yields body angles that transitions robot to desired theta.
        """
        starting_theta = self.theta
        if pivot is None:
            inc = self.theta_rot_step_inc
        else:
            inc = self.theta_step_inc
        iterator = self.wrap_range(starting_theta, desired_theta, inc)
        for theta in iterator:
            self.update_theta(theta, pivot)
            yield np.array([theta, self.alpha])
        # Yield last section.
        theta = self.wrap(desired_theta)
        self.update_theta(theta, pivot)
        yield np.array([theta, self.alpha])
        # Repeat last, for more delay and accuracy.
        if abs(self.alpha) < 1:
            yield np.array([theta, self.alpha])
    
    def update_theta(self, theta:float, pivot: str):
        theta = self.wrap(theta)
        pivot_length = self.specs.pivot_length[self.mode,:]
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        # Update positions of all robots.
        for robot in range(self.specs.n_robot):
            if pivot == "a":
                # posa remains intact.
                # pos is half way across -leg_vect.
                self.pos[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                       -pivot_length[robot]*leg_vect/2)
                # posb is across -leg_vect.
                self.posb[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                             -pivot_length[robot]*leg_vect)
            elif pivot == "b":
                # posb remains intact.
                # pos is half way across leg_vect.
                self.pos[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                       +pivot_length[robot]*leg_vect/2)
                # posb is across leg_vect.
                self.posa[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                             +pivot_length[robot]*leg_vect)
            elif pivot == None:
                # Rotationg around center.
                assert abs(self.alpha) <.01
                # pos remains intact, call reset_state.
                self.reset_state(theta = theta)
                break
            else:
                exc_msg = ("\"pivot\" should be either \"a\" or \"b\""
                           +" or \"None\" for central rotation.")
                raise ValueError(exc_msg)
        # Update theta
        self.theta = theta
    
    def step_mode(self, des_mode: int, des_ang: float, line_up = True):
        """
        Yields body angles that transitions robot to desired mode at
        the given angle.
        @param: des_mode: describes the desired mode to go.
        @param: des_ang: describes the angle that center of each robot
                         will go from start to end of movement.
        @param: line_up: determines if the robots will line_up before
                performing mode_change.
                If False, no line_up will be performed and mode change
                happens from the current states. Thus, des_ang will be
                irrelevant. This is used for test and calibration.
        """
        str_msg = "Mode change can only be started from alpha = 0."
        assert abs(self.alpha) <.01, str_msg
        # Get index of desired mode from the current sequence to
        # determine parameters for tumbling process.
        # Mode index to get tumbling angles and distance.
        if self.mode != des_mode:
            des_mode_index = self.mode_sequence.index(des_mode)
            if line_up:
                theta_start = des_ang-self.specs.mode_rel_ang[des_mode_index]/2
                theta_end = des_ang + self.specs.mode_rel_ang[des_mode_index]/2
            else:
                theta_start = self.theta
                des_ang = theta_start+self.specs.mode_rel_ang[des_mode_index]/2
                theta_end = self.theta+self.specs.mode_rel_ang[des_mode_index]
            # Positions 
            pos_start = self.pos
            pos_delta = (self.specs.mode_rel_length[des_mode_index]*
                         np.array([np.cos(des_ang), np.sin(des_ang)]*
                         self.specs.n_robot))
            pos_end = pos_start + pos_delta
            # Line up the robots based on the mode change angle.
            yield from self.step_theta(theta_start)
            # Lift the robots of pivot point A.
            yield from self.step_alpha(-np.pi/2, self.tumble_step_inc)
            # Update theta and mode, and position.
            theta_end = Controller.wrap(theta_end)
            self.reset_state(pos = pos_end, theta=theta_end, mode=des_mode)
            # Lift down the robots.
            yield from self.step_alpha(0.0, self.tumble_step_inc)
    
    def mode_changing(self, des_mode: int, des_ang: float,
                            last_section = False, line_up = True):
        """
        High level command to change mode.
        @param: des_mode: describes the desired mode to go.
        @param: des_ang: describes the angle that center of each robot
                         will go from start to end of movement.
        @param: line_up: line_up or not before starting.
        """
        for body_ang in self.step_mode(des_mode, des_ang, line_up):
            # Convert body ang to field_ang.
            field_ang = self.angle_body_to_magnet(body_ang)
            # Yield outputs.
            yield field_ang, self.get_state()
        # Line up the robots, if this was last section.
        if last_section is True:
            for body_ang in self.step_theta(des_ang):
                # Convert body ang to field_ang.
                field_ang = self.angle_body_to_magnet(body_ang)
                # Yield outputs.
                yield field_ang, self.get_state()
    
    def step_tumble(self, input_cmd: np.ndarray,
                          last_section = False, line_up = True):
        """
        Yields magnet angles that walks the robot using tumbling mode.
        @param: Numpy array as [distance to walk, theta, mode]
        """
        assert abs(self.alpha) <0.01, "Tumbling should happen at alpha = 0."
        if not line_up:
            # Desired angle command will be overriden in this case.
            input_cmd[1] = Controller.wrap(self.theta - np.pi/2)
        # Modify the distance to walk and theta is necessary.
        if input_cmd[0] < 0:
            input_cmd[0] = -input_cmd[0]
            input_cmd[1]  = Controller.wrap(input_cmd[1] + np.pi)
        # Calculate position update step.
        pos_delta = (self.specs.tumbling_length
                    *np.array([np.cos(input_cmd[1]), np.sin(input_cmd[1])]
                    *self.specs.n_robot))
        # determine rounded number of rotations needed.
        n_tumbling = round(input_cmd[0]/self.specs.tumbling_length)
        # Line up robots for starting with pivot B.
        start_theta = Controller.wrap(input_cmd[1] + np.pi/2)
        yield from self.step_theta(start_theta)
        # Set tumbling parameter.
        alpha_dir = 1
        next_mode={1:self.mode_sequence[(self.specs.n_mode-1)//2],-1:self.mode}
        # Perform tumbling.
        for _ in range(n_tumbling):
            # Lift the robot.
            yield from self.step_alpha(alpha_dir*np.pi/2, self.tumble_step_inc)
            # Update position, theta, and mode
            self.reset_state(pos = self.pos + pos_delta,
                             theta = Controller.wrap(self.theta + np.pi),
                             mode = next_mode[alpha_dir])
            # Take down the robot.
            yield from self.step_alpha(0, self.tumble_step_inc)
            # Update alpha_direction.
            alpha_dir *= -1
        # Line up the robots, if this was last section.
        if last_section is True:
            yield from self.step_theta(input_cmd[1])

    def tumbling(self, input_cmd: np.ndarray,
                       last_section = False, line_up = True):
        """
        High level function for step_tumble.
        @param: Numpy array as [distance to walk, theta, mode]
        """
        for body_ang in self.step_tumble(input_cmd, last_section, line_up):
            # Convert body ang to field_ang.
            field_ang = self.angle_body_to_magnet(body_ang)
            # Yield outputs.
            yield field_ang, self.get_state()

    def rotation(self, theta: float):
        """Rotates robots in place."""
        for body_ang in self.step_theta(theta):
            # Convert body ang to field_ang.
            field_ang = self.angle_body_to_magnet(body_ang)
            # Yield outputs.
            yield field_ang, self.get_state() 

    def pivot_walking(self, theta: float, sweep: float, steps: int,
                                                         last_section = False,
                                                         line_up = True):
        """
        Yields body angles for pivot walking with specified sweep angles
        and number of steps.
        """
        assert steps >= 0, "\"steps\" should be positive integer."
        direction = 1  # 1 is A pivot, -1 is B pivot.
        pivot = {1:"a", -1:"b"}
        # Line of the robot for currect direction.
        if line_up:
            yield from self.step_theta(theta- sweep)
        for _ in range(steps):
            # First pivot around A (lift B) and toggle until completed.
            # Lift the robot by sweep_alpha.
            yield from self.step_alpha(-direction*self.sweep_alpha)
            # Step through theta.
            yield from self.step_theta(theta+direction*sweep,
                                       pivot[direction])
            # Put down the robot.
            yield from self.step_alpha(0.0)
            # Toggle pivot.
            direction *= -1
        if last_section:
            # Line up the robot.
            yield from self.step_theta(theta)

    def pivot_walking_alt(self, theta: float, sweep: float, steps: int,
                                                         last_section = False,
                                                         line_up = True):
        """
        Alternative pivot walking method.
        Yields body angles for pivot walking with specified sweep angles
        and number of steps.
        """
        assert steps >= 0, "\"steps\" should be positive integer."
        direction = 1  # 1 is A pivot, -1 is B pivot.
        pivot = {1:"a", -1:"b"}
        # Line of the robot for currect direction.
        if line_up:
            yield from self.step_theta(theta- sweep)
        for _ in range(steps):
            step_starting_theta = self.theta
            for theta_s, alpha_s in Controller.range_oval(sweep,
                                                      self.sweep_alpha,
                                                      self.pivot_step_inc):
                # Update relted states.
                self.update_alpha(-direction*alpha_s)
                self.update_theta(step_starting_theta
                                  + direction*theta_s,pivot[direction])
                # Yield the values.
                yield np.array([self.theta, self.alpha])
            # Update direction.
            direction *= -1
        if last_section:
            # Line up the robot.
            yield from self.step_theta(theta)
    
    def pivot_walking_walk(self,input_cmd, alternative = True):
        """
        Calls pivot walking and yields field angles and states.
        @param: Numpy array as [n_steps, starting_theta (degrees)]
        """
        assert input_cmd[0] >=0, "n_steps cannot be negative."
        input_cmd[0] = int(input_cmd[0])
        input_cmd[1] = np.deg2rad(input_cmd[1])
        self.reset_state(theta = input_cmd[1])
        # Do pivot walking.
        if alternative:
            for body_ang in self.pivot_walking_alt(input_cmd[1],
                                 self.sweep_theta, input_cmd[0], False, False):
                # Convert body ang to field_ang.
                field_ang = self.angle_body_to_magnet(body_ang)
                # Yield outputs.
                yield field_ang, self.get_state()
        else:
            for body_ang in self.pivot_walking(input_cmd[1],
                                 self.sweep_theta, input_cmd[1], False, False):
                # Convert body ang to field_ang.
                field_ang = self.angle_body_to_magnet(body_ang)
                # Yield outputs.
                yield field_ang, self.get_state()

    def feedforward_walk(self, input_cmd: np.ndarray,
                               last_section = False,
                               alternative = True):
        """
        Generates and yields body angles for pivot walking.
        @param: Numpy array as [distance to walk, theta, mode]
        """
        # Check if the commanded input mode matched the current mode.
        if input_cmd[2] != self.mode:
            exc_msg = "Input is incompatible with current mode."
            raise ValueError(exc_msg)
        # Determine current sweep angle.
        # Get pivot length of leader robot in current mode.
        pivot_length = self.specs.pivot_length[self.mode,0]
        # Calculate maximum distance the leader can travel in one step.
        d_step_max = pivot_length*np.sin(self.sweep_theta)
        # Number of steps takes to do the pivot walk.
        n_steps = int(np.ceil(input_cmd[0]/d_step_max))
        # Compute current sweep angle.
        if n_steps > 0:
            d_step = input_cmd[0]/n_steps
            sweep = np.arcsin(d_step/pivot_length)
            # Do pivot walking.
            if alternative:
                for body_ang in self.pivot_walking_alt(input_cmd[1],sweep,
                                                         n_steps,last_section):
                    # Convert body ang to field_ang.
                    field_ang = self.angle_body_to_magnet(body_ang)
                    # Yield outputs.
                    yield field_ang, self.get_state()
            else:
                for body_ang in self.pivot_walking(input_cmd[1],sweep,
                                                         n_steps,last_section):
                    # Convert body ang to field_ang.
                    field_ang = self.angle_body_to_magnet(body_ang)
                    # Yield outputs.
                    yield field_ang, self.get_state()

    def line_input_compatibility_check(self, input_series: np.ndarray,
                                       alternative = True):
        """
        Checks if an input_series is a compatible sequence.
        @param: 2D array of commands as [distance, angle, mode] for 
        each row.
        """
        # Get states, to reset them after compatibility check.
        states = self.get_state()
        num_sections = input_series.shape[0]
        last_section = False
        # Execute the input step by step.
        # Raise error if not compatible.
        # Reset states to their initial value in any condition.
        try: 
            for section in range(num_sections):
                if section == (num_sections - 1):
                    # If last section, robot will finally line up.
                    last_section = True
                current_input = input_series[section,:]
                current_input_mode = int(current_input[2])
                if current_input_mode < 0:
                    # This is mode change request.
                    cmd_mode = -current_input_mode
                    cmd_ang = current_input[1]
                    for _ in self.mode_changing(cmd_mode, cmd_ang,last_section):
                        pass
                elif current_input_mode == 0:
                    # This is tumbling.
                    for _ in self.tumbling(current_input, last_section):
                        pass
                elif current_input_mode == 999:
                     # This is rotation in place.
                     for _ in self.rotation(current_input[1]):
                         pass
                else: 
                    # This is pivot walking mode.
                    # First check compatibility.
                    if current_input_mode != self.mode:
                        exc_msg = (f"Input series section: {section+1:02d}"
                                +" has incompatible mode")
                        raise ValueError(exc_msg)
                    # If no exception is occured, run the section.
                    for _ in self.feedforward_walk(current_input, last_section,
                                                                  alternative):
                        pass
        finally:
            # Reset the states to its initials.
            self.reset_state(*states)

    def feedforward_line(self, input_series:np.ndarray,
                               alternative = True, has_last_section = True):
        """
        Generates magnetic field angles and state transition of the
        milli-robots, to execute a series of linear input commands
        through feedforward control.
        @param: 2D array of commands as [distance, angle, mode] for 
        each row.

        line_input_compatibility_check method should be run before this
        method to see if the input_series is compatible.
        """
        num_sections = input_series.shape[0]
        last_section = False
        for section in range(num_sections):
            if section == (num_sections - 1):
                # If last section, robot will finally line up.
                last_section = has_last_section
            current_input = input_series[section,:]
            current_input_mode = int(current_input[2])
            if current_input_mode < 0:
                # This is mode change request.
                cmd_mode = -current_input_mode
                cmd_ang = current_input[1]
                yield from self.mode_changing(cmd_mode, cmd_ang, last_section)
            elif current_input_mode == 0:
                # This is tumbling.
                yield from self.tumbling(current_input, last_section)
            elif current_input_mode == 999:
                # This is rotation in place.
                yield from self.rotation(current_input[1])
            else: 
                # This is pivot walking mode.
                yield from self.feedforward_walk(current_input,
                                                      last_section,alternative)

class Pipeline:
    """
    This class manages command mode and commands in those mode.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.__setup()

    def __setup(self):
        cmd = {"idle":   np.array([0.0,0.0,0.0]),
               "server": np.array([0.0,0.0,0.0])}
        self.cmd_mode = "idle"
        self.cmd = cmd
        self.states = None
    
    def set_cmd(self, cmd: np.ndarray):
        self.lock.acquire()
        self.cmd["server"] = cmd
        self.lock.release()

    def set_state(self, states):
        self.lock.acquire()
        self.states = states
        self.lock.release()
    
    def set_idle(self, cmd: np.ndarray):
        self.lock.acquire()
        self.cmd["idle"] = cmd
        self.lock.release()

    def set_cmd_mode(self, cmd_mode):
        self.lock.acquire()
        if cmd_mode in self.cmd.keys():
            self.cmd_mode = cmd_mode
            self.lock.release()
        else:
            print("Invalid cmd_mode. Check cmd.keys().")
    
    def get_cmd_mode(self):
        self.lock.acquire()
        cmd_mode = self.cmd_mode
        self.lock.release()
        return cmd_mode
    
    def get_cmd(self):
        self.lock.acquire()
        cmd = self.cmd[self.cmd_mode]
        states = self.states
        self.lock.release()
        return cmd, states

########## test section ################################################
if __name__ == '__main__':
    specs = model.SwarmSpecs.robo3()
    control = Controller(specs,np.array([0,0,20,0,40,0]),0,1)
    input_series = np.array([[10,0,1],
                             [0,0,1],
                             [12,0,-2],
                             [12*2,np.pi/2,0],
                             [10,0,2],
                             [12,0,0],
                             [0,np.pi/2,1],
                             [0,np.pi/4,1]])
    """ input_series = np.array([[50,0,1],
                                 [50,np.pi/2,1],
                                 [50,np.pi,1],
                                 [50,-np.pi/2,1],
                                 [0,0,1]])
    input_series = input_series = np.tile(input_series, (2,1)) """
    # Check compatibility
    control.line_input_compatibility_check(input_series)
    start = time.time()
    for i in control.feedforward_line(input_series,alternative=True):
        str_msg = (",".join(f"{elem:+07.2f}" for elem in i[0]) + "|"
                  +",".join(f"{elem:+07.2f}" for elem in i[1][0]) + "|"
                  +f"{i[1][1]*180/np.pi:+07.2f},{i[1][2]*180/np.pi:+07.2f}, " 
                  +f"{i[1][3]:01d}")
        print(str_msg)
    end = time.time()
    print(f"Elapsed = {(end-start)*1000:+015.2f}milliseconds")
