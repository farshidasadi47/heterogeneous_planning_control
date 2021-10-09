########################################################################
# This files hold classes and functions that simulates the milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import numpy as np
np.set_printoptions(precision=4, suppress=True)

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
        self.specs = specs
        self.reset_state(position, angle, mode)

    def reset_state(self,position, angle, mode):
        if (position.shape[0]//2 != self.specs.n_robot):
            error_message = """Position does not match number of the robots."""
            raise ValueError(error_message)
        self.position = position
        self.angle = angle
        self.mode = mode

    def update_state(self, u, is_rotation = False):
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
        if is_rotation is False:
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
        # Update states of swarm of milirobots
        self.position = self.position + np.dot(B[mode,:,:],u)
        self.angle = theta
        if mode == 0:
            # If there are rotations, calculate mode after rotations.
            self.mode = int((self.mode + rotations - 1)%(self.specs.n_mode - 1) + 1)
    
    def simulate(self, input_series, position = None,
                 angle = None, mode = None):
        if (input_series.ndim != 2):
            raise ValueError('Input series should be a 2D numpy array')
        if position is None:
            position = self.position
        if angle is None:
            angle = self.angle
        if mode is None:
            mode = self.mode
        
        self.reset_state(position, angle, mode)
        Position = self.position.reshape(-1,1)
        Angle = self.angle
        Mode = self.mode
        Input = np.array([], dtype=float).reshape(input_series.shape[0],0)

        for section in range(input_series.shape[1]):
            current_r = input_series[0,section]
            current_angle = input_series[1,section]
            current_input_mode = input_series[2,section]
            steps = int(current_r//self.specs.rotation_distance)
            rem_r =  current_r%self.specs.rotation_distance
            if current_input_mode == 0:
                is_rotation = True
            else: 
                is_rotation = False
            # Implementing current input section in smaller steps
            for step in range(1,steps+1):
                step_input = np.array([self.specs.rotation_distance,
                                       current_angle, current_input_mode]).T
                Input = np.hstack((Input,step_input.reshape(-1,1)))
                # Applying current step
                self.update_state(step_input[:2], is_rotation)
                Position = np.hstack((Position,self.position.reshape(-1,1)))
                Angle = np.hstack((Angle,self.angle))
                Mode = np.hstack((Mode,self.mode))
            # Implementing remainder of section
            step_input = np.array([rem_r,
                                   current_angle, current_input_mode]).T
            Input = np.hstack((Input,step_input.reshape(-1,1)))

            self.update_state(step_input[:2], is_rotation)
            Position = np.hstack((Position,self.position.reshape(-1,1)))
            Angle = np.hstack((Angle,self.angle))
            Mode = np.hstack((Mode,self.mode))
            # Changing to next mode if needed
            if (section+1 < input_series.shape[1]):
                # If this is not last section.
                if (current_input_mode != input_series[2, section +1] and
                     input_series[2, section +1] != 0):
                    # If there was a mode change, go to next mode.
                    is_rotation = True
                    step_input = np.array([self.specs.rotation_distance,
                                           current_angle, 0]).T
                    Input = np.hstack((Input,step_input.reshape(-1,1)))
                    print(self.position)
                    self.update_state(step_input[:2], is_rotation)
                    print(self.position)
                    Position = np.hstack((Position,
                                          self.position.reshape(-1,1)))
                    Angle = np.hstack((Angle,self.angle))
                    Mode = np.hstack((Mode,self.mode))
        mode_change_index = np.where(Input[2,:-1] != Input[2,1:])[0]+1
        return (Position, Angle, Mode, Input, mode_change_index)


########## test section ################################################
if __name__ == '__main__':
    swarm_specs = SwarmSpecs(np.array([[9,7,5,3],[3,5,7,9],[3,2,6,9]]), 5, 10)
    swarm = Swarm(np.array([0,0,10,0,20,0,30,0]), 0, 1, swarm_specs)
    #input_series = np.array([[100,np.pi/4,1]]).T
    input_series = np.array([[100,np.pi/4,1],
                             [53.2, 4.15, 2],
                             [3.251,5.325,2],
                             [35.214,.354,3],
                             [50,3.24,1]]).T
    sim = swarm.simulate(input_series)
    #print(swarm.specs.pivot_seperation)
    #print(swarm.specs.beta)
    #print(swarm.specs.B)
    #print(swarm.mode)
    #print(swarm.specs.beta)
    print(sim[4])
    print(sim[0][:,np.concatenate(([0],sim[4],[-1]))].T)
    #print(sim[0][:,[19,20,21,22]].T)
    #print(sim[4])
    #print(np.floor(swarm.specs.rotation_distance/swarm.specs.rotation_distance))
    #print(sim[0][:,-1])
    #swarm.reset_state(sim[0][:,-1], sim[1][-1],sim[2][-1])
    #u = np.array([swarm.specs.rotation_distance,swarm.angle,0])
    #swarm.update_state(u[:2],True)
    #print(swarm.position)

