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

import numpy as np
import numpy.matlib
import casadi as ca

from scipy.spatial.transform import Rotation

import model

np.set_printoptions(precision=2, suppress=True)
########## classes and functions #######################################
@dataclass
class Robots:
    pivot_separation: np.ndarray
    rotation_distance: float
    tumbling_distance: float
    def to_list(self):
        return list(vars(self).values())

class ControlModel():
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
        self.theta_step_increment = np.deg2rad(2)
        self.alpha_step_increment = np.deg2rad(4)
        self.rot_step_increment = np.deg2rad(8)
        self.sweep_theta = np.deg2rad(30)  # Max sweep angle
        self.sweep_alpha = np.deg2rad(20)  # alpha sweep limit.

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
        pivot_seperation = self.specs.pivot_seperation[mode,:]
        leg_vect = np.array([0,self.specs.pivot_seperation[mode,0]/2,0])
        self.pos = pos.astype(float)
        # Calculate leg positions.
        self.posa = np.zeros_like(self.pos)
        self.posb = np.zeros_like(self.pos)
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        for robot in range(self.specs.n_robot):
            # a is half way along +y or leg_vect.
            self.posa[2*robot:2*robot+2] = (pivot_seperation[robot]*leg_vect/2
                                            + self.pos[2*robot:2*robot+2])
            # b is half way along -y or -leg_vect
            self.posb[2*robot:2*robot+2] = (-pivot_seperation[robot]*leg_vect/2
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
        # Calculate magnet vectors
        self.rot_increment = 2*np.pi/(n_mode-1)
        magnet_vect_base =  np.array([1,-1,0])/np.sqrt(1+1)
        self.magnet_vect = {}
        for mode in range(1,n_mode):
            self.magnet_vect[mode] = self.rotv(magnet_vect_base,
                                               self.rot_vect,
                                               self.rot_increment*(mode-1))
    
    def angle_body_to_magnet(self, ang: np.ndarray, mode: int):
        """
        Converts (theta, alpha) of robots body, to (theta, alpha) of 
        the robots magnets. The converted value can be used as desired
        orientation of coils magnetic field.
        @param: array composed of theta, and alpha of robot body in
                Radians.
         @type: 1D numpy array.
        """
        # Get magnet vector.
        magnet_vect = self.magnet_vect[mode]
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
    def wrap(angle):
        """Wraps angles between -PI to PI."""
        wrapped = remainder(angle+np.pi, 2*np.pi)
        if wrapped< 0:
            wrapped += 2*np.pi
        wrapped -= np.pi
        return wrapped
    
    @staticmethod
    def wrap_range(from_ang, to_ang, inc = 1):
        """
        Yields a range of wrapped angle that goes from \"from_ang\"
        to \"to_ang\".
        """
        diff = float(ControlModel.wrap(to_ang - from_ang))
        if diff < 0:
            inc *= -1
        diff = ControlModel.round_to_zero(diff)
        to_ang = from_ang + diff
        for ang in np.arange(from_ang, to_ang, inc):
            yield ControlModel.wrap(ang)
  
    # Control related methods
    def step_alpha(self, desired_alpha:float):
        """
        Yields body angles that transitions robot to desired alpha.
        """
        starting_alpha = self.alpha
        iterator = self.wrap_range(starting_alpha, desired_alpha,
                                                     self.alpha_step_increment)
        # Skip the first element to avoid repetition.
        try:
            next(iterator)
        except StopIteration:
            pass
        # Yield the rest.
        for alpha in iterator:
            self.update_alpha(alpha)
            yield np.array([self.theta, alpha])
        # Yield last section.
        alpha = self.wrap(desired_alpha)
        self.update_alpha(alpha)
        yield np.array([self.theta, alpha])

    def update_alpha(self, alpha: float):
        self.alpha = alpha

    def step_theta(self, desired_theta:float, pivot: str = None):
        """
        Yields body angles that transitions robot to desired theta.
        """
        starting_theta = self.theta
        iterator = self.wrap_range(starting_theta, desired_theta,
                                                     self.theta_step_increment)
        # Skip the first element to avoid repetition.
        try:
            next(iterator)
        except StopIteration:
            pass
        # Yield the rest.
        for theta in iterator:
            self.update_theta(theta, pivot)
            yield np.array([theta, self.alpha])
        # Yield last section.
        theta = self.wrap(desired_theta)
        self.update_theta(theta, pivot)
        yield np.array([theta, self.alpha])
    
    def update_theta(self, theta:float, pivot: str):
        theta = self.wrap(theta)
        pivot_seperation = self.specs.pivot_seperation[self.mode,:]
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        # Update positions of all robots.
        for robot in range(self.specs.n_robot):
            if pivot == "a":
                # posa remains intact.
                # pos is half way across -leg_vect.
                self.pos[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                       -pivot_seperation[robot]*leg_vect/2)
                # posb is across -leg_vect.
                self.posb[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                             -pivot_seperation[robot]*leg_vect)
            elif pivot == "b":
                # posb remains intact.
                # pos is half way across leg_vect.
                self.pos[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                       +pivot_seperation[robot]*leg_vect/2)
                # posb is across leg_vect.
                self.posa[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                             +pivot_seperation[robot]*leg_vect)
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

    def step_rotation_field(self, n_rotation: int):
        """
        Yields magnet angles that rotates the robot for one step.
        @param: number of required rotations, negative means backward.
        """
        assert self.alpha == 0.0, "Rotaiton should happen at alpha = 0."
        #assert n_rotation != 0, "There is no zero rotation."
        # Calculate axis of rotation.
        # Robot initial axis of rotation, fixed along body, +Y axis.
        rot_axis_base =  np.array([0,1,0]) 
        # Rotate, theta about +Z, to get current rot_axis.
        rot_axis = self.rotz(rot_axis_base, self.theta)  
        # Calculate starting magnet vector by rotating magnet_vect
        # by current theta.
        start_magnet_vect = self.rotz(self.magnet_vect[self.mode], self.theta)
        # Calculate and yield magnetic vectors to perform rotation.
        # Adjust increment and n_rotation for direction.
        if n_rotation >= 0.0:
            rot_increment = self.rot_increment
            step_increment = self.rot_step_increment
        else:
            n_rotation = -n_rotation
            rot_increment = -self.rot_increment
            step_increment = -self.rot_step_increment
        # Do the rotations and yield values.
        for _ in range(n_rotation):
            for ang in np.arange(0, rot_increment, step_increment)[1:]:
                # First element is ignored to avoid repetition.
                magnet_vect = self.rotv(start_magnet_vect, rot_axis, ang)
                # Update states.
                self.update_rotation_field(step_increment)
                # Convert to spherical and yield.
                yield self.cart_to_sph(magnet_vect)
            # Calculate last one.
            magnet_vect = self.rotv(start_magnet_vect, rot_axis, rot_increment)
            # Update states.
            self.update_rotation_field(rot_increment -ang)
            # Update starting rotation vector and mode.
            start_magnet_vect = magnet_vect
            self.reset_state(mode = self.mode_sequence[1])
            # Convert to spherical and yield.
            yield self.cart_to_sph(magnet_vect)

    def update_rotation_field(self, rot_ang: float):
        # Get how much rotation per current step is done.
        rot_ratio_done = rot_ang/self.rot_increment
        rotation_distance = self.specs.rotation_distance
        n_robot = self.specs.n_robot
        theta = self.theta
        # Update the states.
        for robot in range(n_robot):
            self.pos[2*robot:2*robot+2] += (
                                       np.array([np.cos(theta), np.sin(theta)])
                                       *rot_ratio_done*rotation_distance)
    
    def rotation_walking_field(self, input_cmd: np.ndarray):
        """
        Yields magnet angles thatwalks the robot using rotation mode.
        @param: Numpy array as [distance to walk, theta, mode]
        """
        # determine rounded number of rotations needed.
        n_rotation = round(input_cmd[0]/self.specs.rotation_distance)
        yield from self.step_rotation_field(n_rotation)

    def pivot_walking(self, theta: float, sweep: float, steps: int,
                                                         last_section = False):
        """
        Yields body angles for pivot walking with specified sweep angles
        and number of steps.
        """
        assert steps > 0, "\"steps\" should be positive integer."
        direction = 1  # 1 is A pivot, -1 is B pivot.
        pivot = {1:"a", -1:"b"}
        # Line of the robot for currect direction.
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

    def feedforward_walk(self, input_cmd: np.ndarray, last_section = False):
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
        pivot_length = self.specs.pivot_seperation[self.mode,0]
        # Calculate maximum distance the leader can travel in one step.
        d_step_max = pivot_length*np.sin(self.sweep_theta)
        # Number of steps takes to do the pivot walk.
        n_steps = int(input_cmd[0]//d_step_max) + 1
        # Compute current sweep angle.
        d_step = input_cmd[0]/n_steps
        sweep = np.arcsin(d_step/pivot_length)
        # Do pivot walking.
        yield from self.pivot_walking(input_cmd[1],sweep,n_steps,last_section)

    def line_input_compatibility_check(self, input_series: np.ndarray):
        """
        Checks if an input_series is a compatible sequence.
        @param: 2D array of commands as [distance, angle, mode] for 
        each row.
        """
        # Get states, to reset them after compatibility check.
        states = self.get_state()
        num_sections = input_series.shape[1]
        last_section = False
        # Execute the input step by step.
        # Raise error if not compatible.
        # Reset states to their initial value in any condition.
        try: 
            for section in range(num_sections):
                current_input = input_series[section,:]
                current_input_mode = current_input[2]
                if current_input_mode == 0:
                    # This is rotation mode.
                    # Line up the robots and perform the rotation.
                    for _ in self.step_theta(current_input[1]):
                        pass
                    for _ in self.rotation_walking_field(current_input):
                        pass
                else: 
                    # This is pivot walking mode.
                    # First check compatibility.
                    if current_input_mode != self.mode:
                        exc_msg = (f"Input series section: {section+1:02d}"
                                +" has incompatible mode")
                        raise ValueError(exc_msg)
                    # If no exception is occured, run the section.
                    if section == (num_sections - 1):
                        # If this is last section, robots will line up
                        # in their commanded direction.
                        last_section = True
                    for _ in self.feedforward_walk(current_input,last_section):
                        pass
        finally:
            # Reset the states to its initials.
            self.reset_state(*states)


########## test section ################################################
if __name__ == '__main__':
    #pivot_separation = np.array([[10,9,8,7,6],[9,8,7,6,10],[8,7,6,10,9],[7,6,10,9,8]])
    three_robot = Robots(np.array([[9,8,7],[8,7,9]]), 6.5, 12)
    swarm_specs=model.SwarmSpecs(*three_robot.to_list())

    control = ControlModel(swarm_specs, np.array([0,0,20,0,40,0]),0,1)
    print(control.rotz(control.magnet_vect[1],np.pi/4))
