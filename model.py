########################################################################
# This files hold classes and functions that simulates the milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False

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
        # space boundaries
        self.ubx = 120
        self.uby = 95
        

class Swarm:
    """This class holds current state of swarm of milirobots."""
    
    def __init__(self, position, angle, mode, specs: SwarmSpecs):
        self.specs = specs
        self.reset_state(position, angle, mode)
        self.__plot_colors = None
        self.__plot_markers = None
        self.__simulate_result = None

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

        u[0] = r*np.cos(theta)
        u[1] = r*np.sin(theta)
        # Update states of swarm of milirobots
        self.position = self.position + np.dot(B[mode,:,:],u)
        self.angle = theta
        if mode == 0:
            # If there are rotations, calculate mode after rotations.
            self.mode = int((self.mode + rotations - 1)%(self.specs.n_mode - 1) + 1)
    
    def simulate(self, input_series, position = None,
                 angle = None, mode = None):
        """Simulates the swarm for a given logical series of input."""
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
            current_input_mode = input_series[2,section]
            if current_input_mode == 0:
                is_rotation = True
            else: 
                is_rotation = False
                if current_input_mode != self.mode:
                    print("Input series section: {:02d} has incompatible mode".
                          format(section + 1))
                    break
            current_r = input_series[0,section]
            current_angle = input_series[1,section]
            steps = int(current_r//self.specs.rotation_distance)
            rem_r =  current_r%self.specs.rotation_distance
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
            """ # Changing to next mode if needed
            if (section+1 < input_series.shape[1]):
                # If this is not last section.
                if (current_input_mode != input_series[2, section +1] and
                    input_series[2, section +1] != 0):
                    # If there was a mode change, go to next mode.
                    is_rotation = True
                    step_input = np.array([self.specs.rotation_distance,
                                           current_angle, 0]).T
                    Input = np.hstack((Input,step_input.reshape(-1,1)))
                    self.update_state(step_input[:2], is_rotation)
                    Position = np.hstack((Position,
                                          self.position.reshape(-1,1)))
                    Angle = np.hstack((Angle,self.angle))
                    Mode = np.hstack((Mode,self.mode)) """
        mode_change_index = np.where(Input[2,:-1] != Input[2,1:])[0]+1
        mode_change_index = np.concatenate(([0],mode_change_index,
                                            [Position.shape[1]-1]))
        self.__simulate_result = (Position, Angle, Mode,
                                  Input, mode_change_index)
    
    def simulate_simple(self, input_series):
        """Simulates the swarm for a given simple series of input."""
        if (input_series.ndim != 2):
            raise ValueError('Input series should be a 2D numpy array')
        
        Position = self.position.reshape(-1,1)
        U = input_series[:2.:]
        X = np.zeros_like(U)
        X[:,0] = Position
        
        for i in range(input_series.shape[1]-1):
            mode = input_series[2,i]
            X[:,i+1] = X[:,i] + np.dot(self.specs.B[mode,:,:],U[:,i])
        
        return X, U

    
    def __simplot_set(self, ax, boundary = False):
        """Sets the plot configuration. """
        self.__colors = ['k','r','b','g','m','y','c',]
        self.__markers = ['o','s','P','h','*','+','x','d']
        plt.sca(ax)
        if boundary is True:
            plt.ylim([-self.specs.uby,self.specs.uby])
            plt.xlim([-self.specs.ubx,self.specs.ubx])
        plt.title('Swarm transition')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
    
    def __simplot_plot(self, ax, plot_length, last_section = False):
        """Plots the result of simulation for the given length."""
        tumbling_distance = self.specs.tumbling_distance
        plt.sca(ax)
        (Position, Angle, Mode, Input,
         mode_change_index) = self.__simulate_result
        # Draw initial positions and hold it
        for robot in range(self.specs.n_robot):
            current_mode = Input[2,0].astype(int)
            plt.plot(Position[2*robot,0],
                     Position[2*robot+1,0],
                     color = self.__colors[current_mode],
                     marker = self.__markers[robot],
                     linewidth=1,
                     markerfacecolor='none')
            # Draw circle bounding the robots
            circle = plt.Circle([Position[2*robot,0],
                                 Position[2*robot+1,0]],
                                 radius=tumbling_distance/2,
                                 linestyle='--', linewidth=0.5,
                                 edgecolor='k', facecolor = "None")
            ax.add_patch(circle)
        # Draw the rest
        for robot in range(self.specs.n_robot):
            # Go over all robots.
            length_flag = False
            for section in range(1,mode_change_index.size):
                # Go over all sections based on different input mode.
                start_index = mode_change_index[section - 1]
                end_index = mode_change_index[section]
                if end_index > plot_length-1:
                    # Make sure we do not plot more than plot length.
                    end_index = plot_length-1
                    length_flag = True
                if length_flag == True:
                    # Not exceed plot length.
                    break
                if last_section == False:
                    # If 'last_section' is set to False, all path
                    # will be drawn.
                    current_mode = Input[2,start_index].astype(int)
                    plt.plot(Position[2*robot,start_index:end_index+1],
                             Position[2*robot+1,start_index:end_index+1],
                             color = self.__colors[current_mode],
                             marker = self.__markers[robot],
                             linewidth=1,
                             markerfacecolor='none')
            # Plot last section
            label = "robot: {:1d}".format(robot)
            current_mode = Input[2,start_index].astype(int)
            plt.plot(Position[2*robot,start_index:end_index+1],
                     Position[2*robot+1,start_index:end_index+1],
                     color = self.__colors[current_mode],
                     marker = self.__markers[robot],
                     linewidth=1,
                     label = label,
                     markerfacecolor='none')
            # Draw circle bounding the robots
            circle = plt.Circle([Position[2*robot,end_index],
                                 Position[2*robot+1,end_index]],
                                 radius=tumbling_distance/2,
                                 edgecolor='k', facecolor = "None")
            ax.add_patch(circle)
        # Draw usable space boundaries
        rectangle = plt.Rectangle([-(self.specs.ubx-tumbling_distance/2),
                                   -(self.specs.uby-tumbling_distance/2)],
                                  2*(self.specs.ubx-tumbling_distance/2),
                                  2*(self.specs.uby-tumbling_distance/2),
                                  linestyle='--', linewidth=1,
                                  edgecolor='k', facecolor='none')
        ax.add_patch(rectangle)
        ax.legend(handlelength=0)
        plt.show()

    def simplot(self, input_series, plot_length = 10000,
                position = None, angle = None, mode = None,
                boundary = False, last_section = False):
        """Plots the swarm motion for a given logical series of input.
        """
        if (input_series.ndim != 2):
            raise ValueError('Input series should be a 2D numpy array')
        if position is None:
            position = self.position
        if angle is None:
            angle = self.angle
        if mode is None:
            mode = self.mode
        
        # Update the states
        self.reset_state(position, angle, mode)
        # Simulate the system
        self.simulate(input_series)
        # Set the figure properties
        fig, ax = plt.subplots(constrained_layout=True)
        self.__simplot_set(ax, boundary)
        # plot the figure
        self.__simplot_plot(ax, plot_length, last_section)
        return fig, ax
    
    def __animate(self, i, ax, boundary, last_section):
        ax.clear()
        self.__simplot_set(ax, boundary)
        self.__simplot_plot(ax, i, last_section)

    def simanimation(self,input_series, anim_length = 10000,
                     position = None, angle = None, mode = None,
                     boundary = False, last_section = False):
        """This function produces an animation from swarm transition
        for a given logical input series and specified length."""
        if (input_series.ndim != 2):
            raise ValueError('Input series should be a 2D numpy array')
        if position is None:
            position = self.position
        if angle is None:
            angle = self.angle
        if mode is None:
            mode = self.mode
        # Update the states
        self.reset_state(position, angle, mode)
        # Simulate the system
        self.simulate(input_series)
        anim_length = min(anim_length, self.__simulate_result[0].shape[1])
        # Set the figure properties
        fig, ax = plt.subplots(constrained_layout=True)
        self.__simplot_set(ax, boundary)
        # Animating
        anim = animation.FuncAnimation(fig, self.__animate,
                                       fargs=(ax,boundary,last_section),
                                   interval=250, frames=range(1,anim_length+1))
        return anim

########## test section ################################################
if __name__ == '__main__':
    swarm_specs = SwarmSpecs(np.array([[9,7,5,3],[3,5,7,9],[3,2,6,9]]), 5, 10)
    swarm = Swarm(np.array([0,0,10,0,20,0,30,0]), 0, 1, swarm_specs)
    #input_series = np.array([[100,np.pi/4,1]]).T
    input_series = np.array([[50,np.pi/4,1],
                             [100,-np.pi/2,1],
                             [5,-np.pi/2,0],
                             [20,np.pi,2],
                             [5,np.pi,0],
                             [25,np.pi/2,3],
                             [5,np.pi/2,0],
                             [20,np.pi,1],
                             ]).T
    #sim = swarm.simulate(input_series)
    #print(swarm.specs.pivot_seperation)
    #print(swarm.specs.beta)
    #print(swarm.specs.B)
    #print(swarm.mode)
    #print(swarm.specs.beta)
    #print(sim[4])
    #print(sim[0][:,sim[4]].T)
    #print(sim[3][:,sim[4]].T)
    #print(sim[0][:,[19,20,21,22]].T)
    #print(sim[4])
    #print(np.floor(swarm.specs.rotation_distance/swarm.specs.rotation_distance))
    #print(sim[0][:,-1])
    #swarm.reset_state(sim[0][:,-1], sim[1][-1],sim[2][-1])
    #u = np.array([swarm.specs.rotation_distance,swarm.angle,0])
    #swarm.update_state(u[:2],True)
    #print(swarm.position)
    length = 10000
    #swarm.simplot(input_series,length, boundary=True, last_section=False)
    #print(swarm.__simulate_result[0][:,swarm.__simulate_result[4]].T)
    anim = swarm.simanimation(input_series,length,boundary=True, last_section=True)