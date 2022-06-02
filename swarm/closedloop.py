#%%
########################################################################
# This files hold classes and functions that controls swarm of 
# milirobot system with feedback from camera.
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
    This class holds feedback planning and control for swarm of 
    millirobots. For more info on the algorithms refer to the paper.
    """
    def __init__(self, specs: model.SwarmSpecs,
                       pos: np.ndarray, theta: float, mode: int):
        self.specs = specs
        self.theta_inc = specs.theta_inc
        self.alpha_inc = specs.alpha_inc
        self.rot_inc = specs.rot_inc
        self.pivot_inc = specs.pivot_inc
        self.tumble_inc = specs.tumble_inc
        self.theta_sweep = specs.theta_sweep # Max theta sweep.
        self.alpha_sweep = specs.alpha_sweep # Max alpha sweep.
        self._set_rotation_constants_and_functions()
        self.reset_state(pos, theta, 0, mode)

    def _set_rotation_constants_and_functions(self):
        """
        This function initializes magnets vectors for each mode.
        It also constructs lambda functions for rotating 
        given vectors for given amounnts around certain global axes.
        """
        # Set up rotation functions.
        self.rotx = lambda vect, ang: Rotation.from_euler('x', ang).apply(vect)
        self.roty = lambda vect, ang: Rotation.from_euler('y', ang).apply(vect)
        self.rotz = lambda vect, ang: Rotation.from_euler('z', ang).apply(vect)
        # Rotation around a given axis, angles based on right hand rule.
        self.rotv = (lambda vect, axis, ang: 
                          Rotation.from_rotvec(ang*axis).apply(vect).squeeze())
        # Set magnet vector
        self.magnet_vect = np.array([0.0,-1.0,0.0])

    def reset_state(self, pos: np.ndarray = None, theta: float = None,
                          alpha: float = None, mode: int = None):                        
        # Process input.
        pos = self.pos if pos is None else pos
        theta = self.theta if theta is None else self.wrap(theta)
        alpha = self.alpha if alpha is None else alpha
        mode = self.mode if mode is None else int(mode)
        if (pos.shape[0]//2 != self.specs.n_robot):
            msg = "Position does not match number of the robots."
            raise ValueError(msg)
        #
        pivot_length = self.specs.pivot_length[mode,:]/2
        self.pos = pos.astype(float)
        # Calculate leg positions.
        self.posa = np.zeros_like(self.pos)
        self.posb = np.zeros_like(self.pos)
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        for robot in range(self.specs.n_robot):
            # a is half way along +y or leg_vect.
            self.posa[2*robot:2*robot+2] = (self.pos[2*robot:2*robot+2]
                                           +pivot_length[robot]*leg_vect)
            # b is half way along -y or -leg_vect
            self.posb[2*robot:2*robot+2] = (self.pos[2*robot:2*robot+2]
                                           -pivot_length[robot]*leg_vect)
        self.theta = theta
        self.alpha = alpha
        self.mode = mode
        self.update_mode_sequence(mode)

    def update_mode_sequence(self,mode: int):
        self.mode_sequence = deque(range(1,self.specs.n_mode))
        self.mode_sequence.rotate(-mode+1)
    
    def get_state(self):
        return self.pos,self.theta,self.alpha,self.mode,self.posa,self.posb

    def body2magnet(self, ang):
        """
        Converts (theta, alpha) of robots body, to (theta, alpha) of 
        the robots magnets, to use as desired orientation of the field.
        ----------
        Parameters
        ----------
        ang: array = [theta, alpha]
        ----------
        Returns
        ----------
        magnet: array = [theta_m, alpha_m]
        """
        # Calculate the magnet vetor in cartesian coordinate.
        magnet = self.rotx(self.magnet_vect, ang[1]) # Alpha rotation.
        magnet = self.rotz(magnet, ang[0])           # Theta rotation.
        # Convert to spherical coordinate, [theta_m, alpha_m].
        magnet = self.cart_to_sph(magnet)
        return magnet
    
    @staticmethod
    def cart_to_sph(cartesian):
        """Converts cartesian vector to spherical degrees."""
        # alpha: arctan(z/(x**2 + y**2)**.5)
        alpha = np.degrees(np.arctan2(cartesian[2],
                                      np.linalg.norm(cartesian[:2])))
        # theta: arctan(y/x)
        theta = np.degrees(np.arctan2(cartesian[1], cartesian[0]))
        return [theta, alpha]
    
    @staticmethod
    def frange(start, stop=None, step=None):
        """Float version of python range"""
        # This function is from stackoverflow.
        # if set start=0.0 and step = 1.0 if not specified
        start = float(start)
        if stop == None:
            stop = start + 0.0
            start = 0.0
        if step == None:
            step = 1.0
        if step == 0.0:
            raise ValueError("frange() arg 3 must not be zero.")
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
        diff = Controller.wrap(to_ang - from_ang)
        inc = -inc if diff <0 else inc
        to_ang = from_ang + diff
        for ang in Controller.frange(from_ang, to_ang, inc):
            yield Controller.wrap(ang)
    
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
        # r = f(ang), oval polar coordinate equation.
        fr = lambda ang: (np.tan(sweep_alpha)*np.tan(sweep_theta)
                         /np.sqrt((np.tan(sweep_alpha)*np.cos(ang))**2
                                 +(np.tan(sweep_theta)*np.sin(ang))**2))
        #
        for ang in Controller.frange(0,np.pi,inc):
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

    def step_alpha(self, desired:float, inc = None):
        """
        Yields body angles that transitions robot to desired alpha.
        """
        inc = self.alpha_inc if inc is None else inc
        initial = self.alpha
        for alpha in self.wrap_range(initial, desired, inc):
            self.update_alpha(alpha)
            yield [self.theta, alpha]
        # Yield last section.
        alpha = self.wrap(desired)
        self.update_alpha(alpha)
        yield [self.theta, alpha]
        # Repeat last, for more delay and accuracy.
        if abs(alpha) < 1:
            yield [self.theta, alpha]
    
    def update_alpha(self, alpha: float):
        self.alpha = self.wrap(alpha)

    def step_theta(self, desired:float, pivot: str = None):
        """
        Yields body angles that transitions robot to desired theta.
        """
        initial = self.theta
        inc = self.rot_inc if pivot is None else self.theta_inc
        for theta in self.wrap_range(initial, desired, inc):
            self.update_theta(theta, pivot)
            yield [theta, self.alpha]
        # Yield last section.
        theta = self.wrap(desired)
        self.update_theta(theta, pivot)
        yield [theta, self.alpha]
        # Repeat last, for more delay and accuracy.
        if abs(self.alpha) < 1:
            yield [theta, self.alpha]
    
    def update_theta(self, theta:float, pivot: str = None):
        theta = self.wrap(theta)
        pivot_length = self.specs.pivot_length[self.mode,:]
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        # Update positions of all robots.
        if pivot == 'a':
            # posa is intact.
            for robot in range(self.specs.n_robot):
                # pos is @ -leg_vect/2.
                self.pos[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                              -pivot_length[robot]*leg_vect/2)
                # posb is @ -leg_vect.
                self.posb[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                             -pivot_length[robot]*leg_vect)
        elif pivot == "b":
            # posb remains intact.
            for robot in range(self.specs.n_robot):
                # pos is @ +leg_vect/2.
                self.pos[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                              +pivot_length[robot]*leg_vect/2)
                # posa is @ +leg_vect.
                self.posa[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                               +pivot_length[robot]*leg_vect)
        else:
            # Rotationg around center.
            assert abs(self.alpha) <.01
            # pos remains intact, call reset_state.
            self.reset_state(theta = theta)
        # Update theta
        self.theta = theta


########## test section ################################################
if __name__ == '__main__':
    specs = model.SwarmSpecs.robo3()
    control = Controller(specs,np.array([0,0,20,0,40,0]),0,1)
