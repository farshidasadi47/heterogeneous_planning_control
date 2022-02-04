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
        self.step_increment = np.deg2rad(2)

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
        leg_vect = np.array([0,self.specs.pivot_seperation[mode,0]/2,0])
        self.pos = pos
        self.posa = self.rotz(leg_vect, theta)[:2] + pos[:2]
        self.posb = self.rotz(-leg_vect, theta)[:2]+ pos[:2]
        self.theta = theta
        self.alpha = alpha
        self.mode = mode
        self.update_mode_sequence(mode)
    
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
        self.increment = 2*np.pi/(n_mode-1)
        magnet_vect_base =  np.array([1,-1,0])/np.sqrt(1+1)
        self.magnet_vect = {}
        for mode in range(1,n_mode):
            self.magnet_vect[mode] = self.rotv(magnet_vect_base,
                                               self.rot_vect,
                                               self.increment*(mode-1))
    
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
    def wrap(angle):
        """Wraps angles between -PI to PI."""
        wrapped = remainder(angle+np.pi, 2*np.pi)
        if wrapped< 0:
            wrapped += 2*np.pi
        wrapped -= np.pi
        return wrapped
  
    # Control related methods
    def step_alpha(self, desired_alpha:float):
        """
        Yields body angles that transitions robot to desired alpha.
        """
        starting_alpha = self.alpha
        # Determining increment sign.
        if desired_alpha > starting_alpha:
            step_increment = self.step_increment
        else:
            step_increment = -self.step_increment

        for alpha in np.arange(starting_alpha, desired_alpha, step_increment):
            alpha = self.wrap(alpha)
            self.update_alpha(alpha)
            yield alpha
        alpha = self.wrap(desired_alpha)
        self.update_alpha(alpha)
        yield alpha

    def update_alpha(self, alpha: float):
        self.alpha = alpha

    def step_theta(self, desired_theta:float):
        """
        Yields body angles that transitions robot to desired theta.
        """
        starting_theta = self.theta
        # Determining increment sign.
        if desired_theta > starting_theta:
            step_increment = self.step_increment
        else:
            step_increment = -self.step_increment
        for theta in np.arange(starting_theta, desired_theta, step_increment):
            self.theta = theta
            yield self.theta
        self.theta = desired_theta
        yield self.theta

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
            increment = self.increment
            step_increment = self.step_increment
        else:
            n_rotation = -n_rotation
            increment = -self.increment
            step_increment = -self.step_increment
        # Do the rotations and yield values.
        for _ in range(n_rotation):
            for ang in np.arange(0, increment, step_increment):
                magnet_vect = self.rotv(start_magnet_vect, rot_axis, ang)
                # Convert to spherical and yield.
                yield self.cart_to_sph(magnet_vect)
            # Calculate last one.
            magnet_vect = self.rotv(start_magnet_vect, rot_axis, increment)
            # Convert to spherical.
            magnet_sph = self.cart_to_sph(magnet_vect)
            # Update starting rotation vector and mode.
            start_magnet_vect = magnet_vect
            self.update_state(mode = self.mode_sequence[1])
            # Convert to spherical and yield.
            yield self.cart_to_sph(magnet_vect)
        




########## test section ################################################
if __name__ == '__main__':
    #pivot_separation = np.array([[10,9,8,7,6],[9,8,7,6,10],[8,7,6,10,9],[7,6,10,9,8]])
    three_robot = Robots(np.array([[9,8,7],[8,7,9]]), 6.5, 12)
    swarm_specs=model.SwarmSpecs(*three_robot.to_list())

    control = ControlModel(swarm_specs, np.array([0,0,20,0,40,0]),0,1)
    print(control.rotz(control.magnet_vect[1],np.pi/4))
