#%%
########################################################################
# This files hold classes and functions that simulates the milirobot 
# system without considering mode change.
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

from swarm.model import SwarmSpecs
########## Functions ###################################################
class Swarm:
    """This class holds current state of swarm of milirobots."""
    
    def __init__(self, position, specs: SwarmSpecs):
        self.specs = specs
        self.reset_state(position)
        self._simulate_result = None

    def reset_state(self, position):
        position= np.array(position, dtype= float)
        msg = "Position does not match number of the robots in spec."
        assert (position.shape[0]//2 == self.specs.n_robot), msg
        self.position = position
    
    def update_state(self, u: np.ndarray):
        """
        This function updates the position, angle, and mode of swarm
        of milirobots based on given input.
        ----------
        Parameters
        ----------
        u: numpy.ndarray
           A numpy array as [r,phi]
           r:     Distance to travel
           phi: Angle to travel
           mode: int
              The mode motion is performed.
        """
        n_mode = self.specs.n_mode
        u= np.array(u)
        r = u[0]
        phi = u[1]
        mode = int(u[2])
        # Modify input based on requested motion mode.
        if mode > 0:
            # Pivot walking
            B = self.specs.B[mode,:,:]
        else:
            B = self.specs.B[0,:,:]
        # Convert input to cartesian.
        u[0] = r*np.cos(phi)
        u[1] = r*np.sin(phi)
        # Update states of swarm of milirobots
        self.position = self.position + np.dot(B,u[:-1])

    def simulate(self, input_series, position = None, step_size= None):
        """Simulates the swarm for a given logical series of input."""
        msg = 'Input series should be a 2D numpy array.'
        assert (input_series.ndim == 2), msg
        if position is None:
            position = self.position
        #
        self.reset_state(position)
        cum_position = []
        cum_input = []
        # Execute input section by secton.
        for section in input_series:
            cmd_r= section[0]
            cmd_phi= section[1]
            cmd_mode= int(section[2])
            if cmd_mode>=0:
                # Divide the current section to smaller pieces.
                if step_size is None:
                    steps= np.array([cmd_r])
                else:
                    steps= np.arange(0,cmd_r,step_size)
                    steps= np.append(steps, cmd_r)
                    steps= np.diff(steps)
                    if steps.size == 0: # This lets simulate zero inputs.
                        steps = np.array([0.0])
                # Implementing current input section in smaller steps
                sec_position= [self.position]
                sec_input= []
                for step in steps:
                    step_input = np.array([step,cmd_phi, cmd_mode])
                    # Applying current step
                    self.update_state(step_input)
                    sec_position.append(self.position)
                    sec_input.append(step_input)
                # Add zero step for convinience.
                step_input= step_input.copy()
                step_input[0]= 0.0
                sec_input.append(step_input)
                # Store the section's results.
                cum_position.append(np.array(sec_position,dtype= float))
                cum_input.append(np.array(sec_input,dtype= float))
        self._simulate_result = (cum_position, cum_input)
        return cum_position, cum_input

    def _regroup_sim_result(self, paired= False, n_section= 1):
        """
        This function joins consequtive motion sequence based on having
        same mode consequtively, or a given number of sequences.
        """
        # Get the simulation results.
        cum_position, cum_input = self._simulate_result
        g_position= []
        g_input= []
        # Combine consequtive sections with similar mode if requested.
        if paired:
            # Find start and end indexes.
            mode_seq= [i[0,2] for i in cum_input]
            distinct_mode_idx= []
            i, id_s= 0, 0 
            for i in range(1,len(mode_seq)):
                if mode_seq[i]-mode_seq[i-1]:
                    # If mode has changed.
                    distinct_mode_idx.append((id_s,i-1))
                    id_s= i
            distinct_mode_idx.append((id_s,i))
            # Combine sections.
            for idx_s, idx_e in distinct_mode_idx:
                sec_position= cum_position[idx_s]
                sec_input= cum_input[idx_s]
                for sec_pos, sec_inp in zip(cum_position[idx_s+1:idx_e+1],
                                           cum_input[idx_s+1:idx_e+1]):
                    sec_position= np.vstack((sec_position[0:-1,:], sec_pos))
                    sec_input= np.vstack((sec_input[0:-1,:], sec_inp))
                g_position.append(sec_position)
                g_input.append(sec_input)
            cum_position= g_position
            cum_input= g_input
            g_posiiton= []
            g_input= []
        # Combine each N sections in requested.
        indexes= list(range(0,len(cum_input), n_section)) + [len(cum_input)]
        indexes= [(i,j) for i,j in zip(indexes[:-1],indexes[1:])]
        # Combine sections.
        for idx_s, idx_e in indexes:
            sec_position= cum_position[idx_s]
            sec_input= cum_input[idx_s]
            for sec_pos, sec_inp in zip(cum_position[idx_s+1:idx_e],
                                        cum_input[idx_s+1:idx_e]):
                sec_position= np.vstack((sec_position[0:-1,:], sec_pos))
                sec_input= np.vstack((sec_input[0:-1,:], sec_inp))
            g_position.append(sec_position)
            g_input.append(sec_input)
        return g_position, g_input

def main():
    specs = SwarmSpecs(np.array([[10,5,5],[5,5,10]]), 10)
    #specs = SwarmSpecs.robo3()
    xi = np.array([-20,0,0,0,20,0])
    mode = 1
    swarm = Swarm(xi, specs)
    input_series = np.array([[60,np.pi/2,1],
                             [70,np.pi,1],
                             [70,0,0],
                             [70,0,2],
                             [70,np.pi/2,2],
                             ])
    step_size= 20
    cum_position, cum_input= swarm.simulate(input_series, xi, step_size)
    g_position, g_input= swarm._regroup_sim_result(paired=True, n_section=1)

########## test section ################################################
if __name__ == '__main__':
    main()
