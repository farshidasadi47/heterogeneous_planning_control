########################################################################
# This files hold classes and functions that simulates the milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from dataclasses import dataclass
import numpy as np

########## Classes #####################################################
class SwarmSpecs:
    """This class stores specifications of swarm of milirobots."""
    
    def __init__(self,
                pivot_seperation: np.array,
                rotation_distance,
                tumbling_distance):
        if (pivot_seperation.ndim != 2):
            raise ValueError('pivot_seperation should be a 2D numpy array')
        self.n_mode = pivot_seperation.shape[0] + 1
        self.n_robot = pivot_seperation.shape[1]
        self.pivot_seperation = np.vstack((np.ones(self.n_robot),
                                           pivot_seperation.astype(float)))
        self.rotation_distance = rotation_distance
        self.tumbling_distance = tumbling_distance
        self.beta = np.zeros_like(self.pivot_seperation)
        # beta := pivot_seperation_robot_j/pivot_seperation_robot_1
        for mode in range(self.n_mode):
            self.beta[mode,:] = (self.pivot_seperation[mode,:]
                                 /self.pivot_seperation[mode,0])
        self.B = np.zeros((self.n_mode,2*self.n_robot,2))
        for mode in range(self.n_mode):
            for robot in range(self.n_robot):
                self.B[mode,2*robot:2*(robot+1),:] = (self.beta[mode,robot]
                                                      *np.eye(2))
        

class Swarm:
    """This class holds current state of swarm of milirobots."""
    
    def __init__(self, position, angle, mode, specs: SwarmSpecs):
        if (position.shape[0]//2 != specs.n_robot):
            error_message = """Position does not match number of the robots."""
            raise ValueError(error_message)
        self.position = position
        self.angle = angle
        self.mode = mode
        self.specs = specs
        
    def update_state(self, u, is_rotation = None):
        """This function updates the position, angle, and mode of swarm
        of milirobots based on given input and mode.
        The function receives up to two inputs.
        u: a numpy array as [r,theta]
           r:     distance to travel
           theta: angle to travel
        is_rotation: is an optional input that represents an intention
        to travel in rotation mode.
        """
        u = u.astype(float)
        rotations = 0
        B = self.specs.B
        if is_rotation is None:
            mode = self.mode
        else:
            mode = 0

        r = u[0]
        theta = u[1]
        # determine number of rotations, if any, and then update r
        if mode == 0:
            # If the movement is in rotation mode, r should be
            # an integer multiple of rotation_distance
            rotations = np.floor(r/self.specs.rotation_distance)
            r = rotations*self.specs.rotation_distance

        u[0] = -r*np.sin(theta)
        u[1] = r*np.cos(theta)
        self.position = self.position + np.dot(B[mode,:,:],u)
        self.angle = theta
        if mode == 0:
            # If there are rotations, calculate mode after rotations.
            self.mode = (self.mode + rotations - 1)%(self.specs.n_mode - 1) + 1


########## test section ################################################
if __name__ == '__main__':
    swarm_specs = SwarmSpecs(np.array([[9,7,5,3],[3,5,7,9],[3,2,6,9]]), 5, 10)
    swarm = Swarm(np.array([0,0,1,1,2,2,3,3]), 10, 2.0, swarm_specs)
    print(swarm.specs.pivot_seperation)
    print(swarm.specs.beta)
    print(swarm.specs.B)
    print(swarm.mode)
    swarm.update_state(np.array([25,10]),1)
    print(swarm.mode)