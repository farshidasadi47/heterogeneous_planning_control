########################################################################
# This files hold classes and functions that controls swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from itertools import combinations
from collections import deque

import numpy as np
import numpy.matlib
import casadi as ca

from scipy.spatial.transform import Rotation

import model

np.set_printoptions(precision=2, suppress=True)
########## classes and functions #######################################
class Control():
    """This class holds controlers for pivot walking and rolling of
    swarm of millirobots."""
    def __init__(self, specs: model.SwarmSpecs):
        self.specs = specs
        self.__set_rotation_constants_and_functions()
    
    def __set_rotation_constants_and_functions(self):
        """This function initializes magnets vectors for each mode.
        It also constructs lambda functions for rotating 
        given vectors for given amounnts around certain global axes."""
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
        increment = 2*np.pi/(n_mode-1)
        magnet_vect_base =  np.array([1,-1,0])/np.sqrt(1+1)
        self.magnet_vect = {}
        for mode in range(1,n_mode):
            self.magnet_vect[mode] = self.rotv(magnet_vect_base,
                                               self.rot_vect,
                                               increment*(mode-1))

        



########## test section ################################################
if __name__ == '__main__':
    #pivot_separation = np.array([[10,9,8,7,6],[9,8,7,6,10],[8,7,6,10,9],[7,6,10,9,8]])
    pivot_separation = np.array([[10,9,8,7],[9,8,7,10],[8,7,10,9]])
    #pivot_separation = np.array([[10,9,8],[9,8,10]])
    swarm_specs=model.SwarmSpecs(pivot_separation, 5, 10)

    control = Control(swarm_specs)
    print(control.rotz(control.magnet_vect[1],np.pi/4))



