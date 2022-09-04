#%%
########################################################################
# This files hold classes and functions that simulates the milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
from itertools import combinations
from collections import deque
from math import remainder

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False

########## Functions ###################################################
def wrap(angle):
    """Wraps angles between -PI to PI."""
    wrapped = remainder(angle+np.pi, 2*np.pi)
    if wrapped< 0:
        wrapped += 2*np.pi
    wrapped -= np.pi
    return wrapped

def define_colors(self):
    self._colors = {'k':(  0,  0,  0), 'r':(  0,  0,255), 'b':(255,  0,  0),
                    'g':(  0,255,  0), 'm':(255,  0,255), 'w':(255,255,255),
                    'y':(  0,255,255), 'c':(255,255,  0)}
    self._markers = ['o','s','P','h','*','+','x','d']
########## Classes #####################################################
class SwarmSpecs:
    """This class stores specifications of swarm of milirobots."""
    
    def __init__(self, pivot_length: np.array, tumbling_length,*,
                      theta_inc = np.deg2rad(5), alpha_inc = np.deg2rad(5),
                      rot_inc = np.deg2rad(5), pivot_inc = np.deg2rad(5),
                      tumble_inc = np.deg2rad(2), theta_sweep = np.deg2rad(30),
                      alpha_sweep = np.deg2rad(30)):
        msg = '\"pivot_length\" should be a 2D numpy array.'
        assert pivot_length.ndim == 2, msg
        msg = "Robots should have even number of sides."
        assert pivot_length.shape[0]%2 == 0, msg
        self.n_mode = pivot_length.shape[0] + 1
        self.n_robot = pivot_length.shape[1]
        self.pivot_length = np.vstack((np.ones(self.n_robot)*tumbling_length,
                                       pivot_length.astype(float)))
        self.tumbling_length = tumbling_length
        self.beta = np.zeros_like(self.pivot_length)
        # beta := pivot_seperation_robot_j/pivot_seperation_robot_1
        for mode in range(self.n_mode):
            self.beta[mode,:] = (self.pivot_length[mode,:]
                                 /self.pivot_length[mode,0])
        self.B = np.zeros((self.n_mode,2*self.n_robot,2))
        for mode in range(self.n_mode):
            for robot in range(self.n_robot):
                self.B[mode,2*robot:2*(robot+1),:] = (self.beta[mode,robot]
                                                      *np.eye(2))
        # Angles and distances related to mode changes
        self.rot_inc = 2*np.pi/(self.n_mode-1)
        self.mode_rel_ang = []
        # Angles to enter modes when lifted on A.
        # Indexes are relative to current mode of the robot.
        for i in range(self.n_mode-1):
            angle = i*self.rot_inc
            self.mode_rel_ang.append(angle)
        # Mode change distances relative to current mode.
        self.mode_rel_length = [0]
        for i in range(1,self.n_mode-1):
            dist = (self.tumbling_length
                   *np.sqrt(2-2*np.cos(self.mode_rel_ang[i])))/2
            self.mode_rel_length.append(dist)
        # Space boundaries
        self.ubx = 115
        self.uby = 90
        self.lbx = -self.ubx
        self.lby = -self.uby
        self.rcoil = 90
        # Some parameters related to planning
        self.robot_pairs = list(combinations(range(self.n_robot),2))
        self.d_min = 20#self.tumbling_length*1.5
        # Adjusted space boundaries for planning.
        self.ubsx = self.ubx-self.tumbling_length*1.5
        self.lbsx = -self.ubsx
        self.ubsy = self.uby - self.tumbling_length*1.1
        self.lbsy = -self.ubsy
        self.rscoil = self.rcoil - self.tumbling_length
        # Plotting and vision markers.
        define_colors(self)
        self._colors = list(self._colors.keys())
        # Experimental execution parameters.
        self.theta_inc = theta_inc
        self.alpha_inc = alpha_inc
        self.rot_inc = rot_inc
        self.pivot_inc = pivot_inc
        self.tumble_inc = tumble_inc
        self.theta_sweep = theta_sweep
        self.alpha_sweep = alpha_sweep
        self.x_tol = 3.0
        self.bc_tol = 0*2.0
        # Drawing latters
        self._set_letters()

    def _set_letters(self):
        chars3= dict()
        chars3['*']= {'poses': [-40,  0,   0,  0,  40,  0],
                      'shape': [0,1, 1,2, 999,999],
                      'steps': 3}
        chars3['b']= {'poses': [+40,+40,   0,  0,   0,+40],
                      'shape': [0,2, 1,2, 999,999],
                      'steps': 3}
        chars3['c']= {'poses': [-40,  0, -40, 40,   0,  0],
                      'shape': [0,1, 0,2, 999,999],
                      'steps': 3}
        chars3['d']= {'poses': [  0,  0,   0,-40, -40,-40],
                      'shape': [0,1, 1,2, 999,999],
                      'steps': 3}
        chars3['e']= {'poses': [+40,-40,   0,  0,  40,  0],
                      'shape': [0,2, 1,2, 999,999],
                      'steps': 3}
        chars3['f']= {'poses': [  0,  0, +40,  0, -40,  0],
                      'shape': [0,1, 0,2, 999,999],
                      'steps': 3}
        #
        chars4= dict()
        chars4['*']= {'poses': [-60,  0, -20,  0,  20,  0,  60,  0],
                      'shape': [0,1, 1,2, 2,3, 999,999],
                      'steps': 3}
        chars4['1']= {'poses': [  0, 60,   0, 20,   0,-20,   0,-60],
                      'shape': [0,1, 1,2, 2,3, 999,999],
                      'steps': 3}
        chars4['7']= {'poses': [-30, 40, -30,-40,   0,  0,  30, 40],
                      'shape': [0,3, 1,2, 2,3, 999,999],
                      'steps': 3}
        chars4['D']= {'poses': [-30, 40, -30,-40,  30, 20,  30,-20],
                      'shape': [0,1, 0,2, 1,3, 2,3],
                      'steps': 3}
        chars4['I']= {'poses': [  0, 60,   0, 20,   0,-20,   0,-60],
                      'shape': [0,1, 1,2, 2,3, 999,999],
                      'steps': 3}
        chars4['J']= {'poses': [-30,-20,   0,-40,  30, 40,  30,-20],
                      'shape': [0,1, 1,3, 2,3, 999,999],
                      'steps': 3}
        chars4['L']= {'poses': [-30, 40, -30,  0, -30,-40,  30,-40],
                      'shape': [0,1, 1,2, 2,3, 999,999],
                      'steps': 3}
        chars4['N']= {'poses': [-30, 40, -30,-40,  30, 40,  30,-40],
                      'shape': [0,1, 0,3, 2,3, 999,999],
                      'steps': 3}
        chars4['T']= {'poses': [-40, 40,   0, 40,   0,-40,  40, 40],
                      'shape': [0,1, 1,2, 1,3, 999,999]}
        chars4['V']= {'poses': [-30, 40, -15,  0,   0,-40,  30, 40],
                      'shape': [0,1, 1,2, 2,3, 999,999],
                      'steps': 3}
        chars4['Y']= {'poses': [-30, 40,   0,  0,   0,-40,  30, 40],
                      'shape': [0,1, 1,2, 1,3, 999,999],
                      'steps': 3}
        chars4['Z']= {'poses': [-20, 40, -30,-40,  30, 40,  30,-40],
                      'shape': [0,2, 1,2, 1,3, 999,999],
                      'steps': 3}
        #
        chars5= dict()
        chars5['*']= {'poses': [-80,  0, -40,  0,   0,  0,  40,  0,  80,  0],
                      'shape': [0,1, 1,2, 2,3, 3,4, 4,5],
                      'steps': 3}
        chars5['A']= {'poses': [-30,-40, -15,  0,   0, 40,  15,  0,  30,-40],
                      'shape': [0,1, 1,2, 1,3, 2,3, 3,4],
                      'steps': 3}
        chars5['B']= {'poses': [-30, 40, -30, -40,  0,  0,  30, 20,  30,-20],
                      'shape': [0,1, 0,3, 1,4, 2,3, 2,4],
                      'steps': 3}
        chars5['C']= {'poses': [-30, 20,   0, 40,   0,-40,  30, 20,  30,-20],
                      'shape': [0,1, 0,2, 1,3, 2,4, 999,999],
                      'steps': 3}
        chars5['D']= {'poses': [-30, 40, -30,-40,  10, 40,  10,-40,  30,  0],
                      'shape': [0,1, 0,2, 1,3, 2,4, 3,4],
                      'steps': 3}
        chars5['F']= {'poses': [-30, 40, -30,  0, -30,-40,  10,  0,  30, 40],
                      'shape': [0,1, 0,4, 1,2, 1,3, 999,999],
                      'steps': 3}
        chars5['I']= {'poses': [  0, 70,   0, 35,   0,  0,   0,-35,   0,-70],
                      'shape': [0,1, 1,2, 2,3, 3,4, 999,999],
                      'steps': 3}
        chars5['J']= {'poses': [-30,-20,   0,-40,  30, 50,  30, 15,  30,-20],
                      'shape': [0,1, 1,4, 2,3, 3,4, 999,999],
                      'steps': 3}
        chars5['K']= {'poses': [-30, 40, -30,  0, -30,-40,  10, 40,  30,-40],
                      'shape': [0,1, 1,2, 1,3, 1,4, 999,999],
                      'steps': 3}
        chars5['L']= {'poses': [-35, 40, -35,  0, -35,-40,   0,-40,  35,-40],
                      'shape': [0,1, 1,2, 2,3, 3,4, 999,999],
                      'steps': 3}
        chars5['M']= {'poses': [-20, 40, -30,-40,   0,  0,  30, 40,  20,-40],
                      'shape': [0,1, 0,2, 2,3, 3,4, 999,999],
                      'steps': 3}
        chars5['N']= {'poses': [-30, 40, -30,-40,   0,  0,  30, 40,  30,-40],
                      'shape': [0,1, 0,2, 2,4, 3,4, 999,999],
                      'steps': 3}
        chars5['O']= {'poses': [-30, 20, -30,-40,   0, 40,  30, 40, 30,-40],
                      'shape': [0,1, 0,2, 1,4, 2,3, 3,4],
                      'steps': 3}
        chars5['P']= {'poses': [-30, 40, -30,  0, -30,-40,  30, 40,  30,  0],
                      'shape': [0,1, 0,3, 1,2, 1,4, 3,4],
                      'steps': 3}
        chars5['R']= {'poses': [-30, 40, -30,  0, -30,-40,  30, 20,  30,-40],
                      'shape': [0,1, 0,3, 1,2, 1,3, 1,4],
                      'steps': 3}
        chars5['S']= {'poses': [-30, 20, -30,-40,   0, 40,  30, 20,  30,-40],
                      'shape': [0,2, 0,4, 1,4, 2,3, 999,999],
                      'steps': 3}
        chars5['T']= {'poses': [-40, 40,   0, 40,  40, 40,   0,  0, -40,  0],
                      'shape': [0,1, 1,2, 1,4, 2,3, 999,999],
                      'steps': 3}
        chars5['U']= {'poses': [-30, 40, -30,-20,   0,-40,  30, 40,  30,-20],
                      'shape': [0,1, 1,2, 2,4, 3,4, 999,999],
                      'steps': 3}
        chars5['V']= {'poses': [-30, 40, -15,  0,   0,-40,  15,  0,  30, 40],
                      'shape': [0,1, 1,2, 2,3, 3,4, 999,999],
                      'steps': 3}
        chars5['W']= {'poses': [-30, 40, -20,-40,   0,  0,  20, 40,  30,-40],
                      'shape': [0,1, 1,2, 2,4, 3,4, 999,999],
                      'steps': 3}
        chars5['X']= {'poses': [-30, 40, -30,-40,   0,  0,  30, 40,  30,-40],
                      'shape': [0,2, 1,2, 2,3, 2,4, 999,999],
                      'steps': 3}
        chars5['Z']= {'poses': [-30, 40, -30,-40, -10,  0,  20, 40,  30,-40],
                      'shape': [0,3, 1,2, 1,4, 2,3, 999,999],
                      'steps': 3}
        #
        chars= {3: chars3, 4:chars4, 5: chars5}
        self.chars= chars.get(self.n_robot, 3)
        for char_dic in self.chars.values():
            char_dic['poses']= np.array(char_dic['poses'],dtype= float)

    def get_letter(self, char, ang = 0, roll= 0):
        chars= self.chars
        scale = 1.0
        roll= roll%self.n_robot
        ang = np.deg2rad(ang)
        poses, shape, steps = chars.get(char, chars['*']).values()
        poses = scale*poses[:2*self.n_robot]
        # Rolling robots position in the pattern
        poses= np.roll(poses, 2*roll)
        shape= [(elem+roll)%self.n_robot for elem in shape]
        # Rotation
        rot = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        poses = np.dot(rot,poses.reshape(-1,2).T).T.flatten()
        return poses, shape, steps

    @classmethod
    def robo3(cls):
        pivot_length = np.array([[9,7,5],[7,5,9]])
        return cls(pivot_length, 10)
    
    @classmethod
    def robo4(cls):
        pivot_length = np.array([[9,7,5,3],[7,5,3,9],[5,3,9,7],[3,9,7,5]])
        return cls(pivot_length,10)

    @classmethod
    def robo5(cls):
        pivot_length = np.array([[11,9,7,5,3],[9,7,5,3,11],
                                 [7,5,3,11,9],[5,3,11,9,7]])
        return cls(pivot_length,10)
    
    @classmethod
    def robo3p(cls):
        pivot_length = np.array([[7.78,5.85,4.49],[6.37,4.39,7.46]])
        #pivot_length = np.array([[8,5,5.0],[5,8,5.0]])
        return cls(pivot_length,11.40)
    
    @classmethod
    def robo4p(cls):
        pivot_length = np.array([[7.15,4.66,4.73,4.64],
                                 [4.63,4.64,4.68,6.98],
                                 [4.70,4.66,6.92,4.64],
                                 [4.63,7.01,4.69,4.62]])
        return cls(pivot_length,13.63)

    @classmethod
    def robo5p(cls):
        pivot_length = np.array([[6.36,4.76,4.67,4.69,6.31],
                                 [4.66,4.73,4.72,6.71,6.31],
                                 [4.59,4.68,6.49,6.70,4.70],
                                 [4.70,6.71,6.56,4.81,4.80]])
        return cls(pivot_length,14.16)
        """ pivot_length= np.array([[7.17,4.84,4.72,4.80,6.99],
                                [4.80,4.74,4.72,7.07,7.00],
                                [4.72,4.76,7.17,7.03,4.71],
                                [4.77,7.03,7.27,4.75,4.72]]) """
        """ pivot_length= np.array([[7.17,4.80,4.82,4.84,7.00],
                                [4.80,4.72,4.83,7.16,7.10],
                                [4.72,4.72,7.15,7.08,4.68],
                                [4.77,7.22,7.22,4.77,4.65]])
        return cls(pivot_length, 13.78) """

    @classmethod
    def robo(cls, n_robot):
        robots = {3: "robo3p", 4: "robo4p", 5: "robo5p"}
        return getattr(cls, robots.get(n_robot, "robo3p"))()
    
class Swarm:
    """This class holds current state of swarm of milirobots."""
    
    def __init__(self, position, angle, mode, specs: SwarmSpecs):
        self.specs = specs
        self.reset_state(position, angle, mode)
        self._simulate_result = None

    def reset_state(self, position, angle, mode):
        msg = "Position does not match number of the robots in spec."
        assert (position.shape[0]//2 == self.specs.n_robot), msg
        self.position = position
        self.angle = angle
        self.mode = mode
        self.update_mode_sequence(mode)
    
    def update_mode_sequence(self,mode: int):
        self.mode_sequence = deque(range(1,self.specs.n_mode))
        self.mode_sequence.rotate(-mode+1)

    def update_state(self, u: np.ndarray, mode: int):
        """
        This function updates the position, angle, and mode of swarm
        of milirobots based on given input.
        ----------
        Parameters
        ----------
        u: numpy.ndarray
           A numpy array as [r,theta]
           r:     Distance to travel
           theta: Angle to travel
        mode: int
              The mode motion is performed.
        """
        n_mode = self.specs.n_mode
        u = u.astype(float)
        r = u[0]
        theta = u[1]
        mode = int(mode)
        # Modify input based on requested motion mode.
        if mode < 0:
            # Mode change, r is irrelevant in this case.
            rel_mode_index = self.mode_sequence.index(-mode)
            r = self.specs.mode_rel_length[rel_mode_index]
            B = self.specs.B[0,:,:]
            self.update_mode_sequence(-mode)
            self.mode = -mode
        elif mode == 0:
            # Tumbling. Modify based on tumbling distance.
            rotations = np.round(r/self.specs.tumbling_length).astype(int)
            r = rotations*self.specs.tumbling_length
            B = self.specs.B[mode,:,:]
            # Update mode if needed.
            if rotations%2:
                self.update_mode_sequence(self.mode_sequence[n_mode//2])
                self.mode = self.mode_sequence[0]
        else:
            # Pivot walking
            B = self.specs.B[mode,:,:]
        # Convert input to cartesian.
        u[0] = r*np.cos(theta)
        u[1] = r*np.sin(theta)
        # Update states of swarm of milirobots
        self.position = self.position + np.dot(B,u)
        self.angle = theta

    def simulate(self, input_series, position = None,
                 angle = None, mode = None):
        """Simulates the swarm for a given logical series of input."""
        msg = 'Input series should be a 2D numpy array.'
        assert (input_series.ndim == 2), msg
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
        # Execute input section by secton.
        for section in range(input_series.shape[1]):
            cmd_mode = int(input_series[2,section])
            cmd_r = input_series[0,section]
            cmd_angle = input_series[1,section]
            # Determine stepping parameters.
            if cmd_mode < 0:
                # Mode_change. Done in one step. The r is irrelevant.
                steps = np.array([cmd_r])
            else:
                if cmd_mode>0 and cmd_mode != self.mode:
                    print(f"Input series section: "
                         +f"{section+1:02d} has incompatible mode.")
                    break
                steps = np.arange(0,cmd_r,self.specs.tumbling_length)
                steps = np.append(steps, cmd_r)
                steps = np.diff(steps)
                if steps.size == 0: # This lets simulate zero inputs.
                    steps = np.array([0])
            # Implementing current input section in smaller steps
            for step in steps:
                step_input = np.array([step,cmd_angle, cmd_mode]).T
                Input = np.hstack((Input,step_input.reshape(-1,1)))
                # Applying current step
                self.update_state(step_input[:2], step_input[2])
                Position = np.hstack((Position,self.position.reshape(-1,1)))
                Angle = np.hstack((Angle,self.angle))
                Mode = np.hstack((Mode,self.mode))
        # Find the indexes where mode change happened.
        mode_change_index = np.where(Input[2,:-1] != Input[2,1:])[0]+1
        mode_change_index = np.concatenate(([0],mode_change_index,
                                            [Position.shape[1]-1]))
        self._simulate_result = (Position, Angle, Mode,
                                  Input, mode_change_index)

    def _simplot_set(self, ax, boundary = False):
        """Sets the plot configuration. """
        if boundary is True:
            ax.set_ylim([-self.specs.uby,self.specs.uby])
            ax.set_xlim([-self.specs.ubx,self.specs.ubx])
        ax.set_title('Swarm transition')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_aspect('equal', adjustable='box')
        ax.grid()

    def _simplot_plot(self, ax, plot_length, last_section = False):
        """Plots the result of simulation for the given length."""
        tumbling_length = self.specs.tumbling_length
        # Geometry of robot symbols.
        aspect_ratio = 2
        h = np.sqrt(tumbling_length**2/(aspect_ratio**2+1))
        w = aspect_ratio*h
        # Get the simulation results.
        (Position, Angle, Mode, Input,
         mode_change_index) = self._simulate_result
        # Draw initial positions and hold it.
        for robot in range(self.specs.n_robot):
            cmd_mode = Input[2,0].astype(int)
            if cmd_mode <0:
                cmd_mode = 0
            cmd_ang = Input[1,0] - np.pi/2
            ax.plot(Position[2*robot,0],
                    Position[2*robot+1,0],
                    color = self.specs._colors[cmd_mode],
                    marker = self.specs._markers[robot],
                    linewidth=1,
                    markerfacecolor='none')
            # Draw a rectangle as the robot.
            x = -(w*np.cos(cmd_ang) - h*np.sin(cmd_ang))/2
            y = -(w*np.sin(cmd_ang) + h*np.cos(cmd_ang))/2
            rect = plt.Rectangle([Position[2*robot,0] + x,
                                 Position[2*robot+1,0] + y] ,
                                 width = w,
                                 height = h,
                                 angle = np.rad2deg(cmd_ang),
                                 linestyle='--', linewidth=0.5,
                                 edgecolor='k', facecolor = "None")
            # Draw circle bounding the robots
            circle = plt.Circle([Position[2*robot,0],
                                 Position[2*robot+1,0]],
                                 #radius=tumbling_length/2,
                                 radius=self.specs.d_min/2,
                                 linestyle='--', linewidth=0.5,
                                 edgecolor='k', facecolor = "None")
            ax.add_patch(rect)
            ax.add_patch(circle)
        # Draw the rest till reacing given plot length.
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
                    cmd_mode = Input[2,start_index].astype(int)
                    if cmd_mode <0:
                        cmd_mode = 0
                    ax.plot(Position[2*robot,start_index:end_index+1],
                            Position[2*robot+1,start_index:end_index+1],
                            color = self.specs._colors[cmd_mode],
                            marker = self.specs._markers[robot],
                            linewidth=1,
                            markerfacecolor='none')
            # Plot last section
            label = "robot: {:1d}".format(robot)
            cmd_mode = Input[2,start_index].astype(int)
            if cmd_mode <0:
                cmd_mode = 0
            cmd_ang = Input[1,min(end_index, Input.shape[1]-1)] - np.pi/2
            ax.plot(Position[2*robot,start_index:end_index+1],
                    Position[2*robot+1,start_index:end_index+1],
                    color = self.specs._colors[cmd_mode],
                    marker = self.specs._markers[robot],
                    linewidth=1,
                    label = label,
                    markerfacecolor='none')
            # Draw a rectangle as the robot.
            x = -(w*np.cos(cmd_ang) - h*np.sin(cmd_ang))/2
            y = -(w*np.sin(cmd_ang) + h*np.cos(cmd_ang))/2
            rect = plt.Rectangle([Position[2*robot,end_index] + x,
                                 Position[2*robot+1,end_index] + y] ,
                                 width = w,
                                 height = h,
                                 angle = np.rad2deg(cmd_ang),
                                 edgecolor='k', facecolor = "None")
            # Draw circle bounding the robots
            circle = plt.Circle([Position[2*robot,end_index],
                                 Position[2*robot+1,end_index]],
                                 #radius=tumbling_length/2,
                                 radius=self.specs.d_min/2,
                                 edgecolor='k', facecolor = "None")
            ax.add_patch(rect)
            ax.add_patch(circle)
        # Draw usable space boundaries
        rectangle = plt.Rectangle([-(self.specs.ubx-tumbling_length/2),
                                   -(self.specs.uby-tumbling_length/2)],
                                  2*(self.specs.ubx-tumbling_length/2),
                                  2*(self.specs.uby-tumbling_length/2),
                                  linestyle='--', linewidth=1,
                                  edgecolor='k', facecolor='none')
        coil = plt.Circle((0,0),radius=self.specs.rcoil,
                                linestyle='--', linewidth=1,
                                edgecolor='k', facecolor='none')
        ax.add_patch(rectangle)
        ax.add_patch(coil)
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
        self._simplot_set(ax, boundary)
        # plot the figure
        self._simplot_plot(ax, plot_length, last_section)
        return fig, ax
    
    def _animate(self, i, ax, boundary, last_section):
        ax.clear()
        self._simplot_set(ax, boundary)
        self._simplot_plot(ax, i, last_section)
        return ax

    def simanimation(self,input_series, anim_length = 10000,
                     position = None, angle = None, mode = None,
                     boundary = False, last_section = False, save = False):
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
        anim_length = min(anim_length, self._simulate_result[0].shape[1])
        # Set the figure properties
        fig, ax = plt.subplots(constrained_layout=True)
        self._simplot_set(ax, boundary)
        # Animating
        anim = animation.FuncAnimation(fig, self._animate,
                                       fargs=(ax,boundary,last_section),
                                   interval=250, frames=range(1,anim_length+1))
        # Saving animation.
        if save:
            # Set file name for saving animation.
            index_for_saving = 1
            anim_name = f"sim_anim_{index_for_saving:02d}.gif"
            anim_directory = os.path.join(os.getcwd(), "result_sim")
            # If the directory does not exist, make one.
            if not os.path.exists(anim_directory):
                os.mkdir(anim_directory)
            anim_path = os.path.join(anim_directory,anim_name)
            # Check if the current file name exists in the directory.
            while os.path.exists(anim_path):
                # Increase file number index until no file with such
                # name exists.
                index_for_saving += 1
                anim_name = f"sim_anim_{index_for_saving:02d}.gif"
                anim_path = os.path.join(anim_directory,anim_name)
            anim.save(anim_path, fps = 4)
        # To ensure the animation is shown.
        plt.show()
        return anim

def main3p():
    specs = SwarmSpecs.robo3p()
    xi = np.array([-20,0,0,0,20,0])
    mode = 1
    swarm = Swarm(xi, 0, mode, specs)
    input_series = np.array([[70,np.pi/2,1],
                             [70,-3*np.pi/4,1],
                             [10,-np.pi/4,-2],
                             [50,-np.pi/2,2],
                             [50,np.pi/4,2],
                             [50,np.pi/4,0],
                             [10,np.pi/4,-1],
                             ]).T
    #swarm.simulate(input_series)
    #print(swarm.specs.pivot_length)
    #print(swarm.specs.beta)
    #print(swarm.specs.B)
    #print(swarm.mode)
    #print(swarm.specs.beta)
    length = 10000
    swarm.simplot(input_series,length, boundary=False, last_section=False)
    #print(swarm._simulate_result[0][:,swarm._simulate_result[4]].T)
    #anim = swarm.simanimation(input_series,length,boundary=False, last_section=True, save = False)
    #sim = swarm._simulate_result # (Position, Angle, Mode,Input, mode_change_index)
    #print(sim[0].T)
    #print(sim[1])
    #print(sim[2])
    #print(sim[3].shape)
    #print(sim[4])

def main4p():
    specs = SwarmSpecs.robo4p()
    xi = np.array([-60,0,-20,0,20,0,60,0],dtype= float)
    mode = 1
    swarm = Swarm(xi, 0, mode, specs)
    input_series = np.array([[4*14,-np.pi/2,0],
                             [100,np.pi/2,1],
                             [10,0,-2],
                             [4*14,-np.pi/2,0],
                             [50,np.pi*2/4,2],
                             [10,np.pi,-3],
                             [50,-np.pi/2,3],
                             [10,0,-4],
                             [50,np.pi/2,4],
                            ]).T
    length = 10000
    swarm.simplot(input_series,length, boundary=False, last_section=False)
    #anim = swarm.simanimation(input_series,length,boundary=False, last_section=True, save = False)

def main5p():
    specs = SwarmSpecs.robo5p()
    xi = np.array([-60,0,-30,0,0,0,30,0,60,0],dtype= float)
    mode = 1
    swarm = Swarm(xi, 0, mode, specs)
    input_series = np.array([[4*14,-np.pi/2,0],
                             [100,np.pi/2,1],
                             [10,0,-2],
                             [6*14,-np.pi/2,0],
                             [50,np.pi*3/4,2],
                             [10,0,-3],
                             [50,0,3],
                             [10,np.pi/2,-4],
                             [50,np.pi*4/4,4],
                            ]).T
    length = 10000
    swarm.simplot(input_series,length, boundary=False, last_section=False)
    #anim = swarm.simanimation(input_series,length,boundary=False, last_section=True, save = False)
########## test section ################################################
if __name__ == '__main__':
    main4p()
