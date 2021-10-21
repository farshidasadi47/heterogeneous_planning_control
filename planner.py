########################################################################
# This files hold classes and functions that plans swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from itertools import combinations
from collections import deque

import numpy as np
import casadi as ca

import model
########## classes and functions #######################################
class Planner():
    """This class contains objects and methods for planning a path
    for a given swarm of milirobots."""

    def __init__(self, swarm: model.Swarm, n_inner = 2, n_outer = 2):
        self.swarm = swarm
        self.robot_combinations = self.__set_robot_combinations()
        self.mode_sequence = self.__set_mode_sequence()
        self.x_final = None
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.ub_space_x = 105
        self.lb_space_x = -105
        self.ub_space_y = 85
        self.lb_space_y = -85
        self.X, self.U, self.P = self.__construct_vars()
    
    def set_space_limit(self, ubx, lbx, uby, lby):
        """Sets space boundary limits and updates the bands in
        optimization problem."""
        self.ub_space_x = ubx
        self.lb_space_x = lbx
        self.ub_space_y = uby
        self.lb_space_y = lby
    
    def __set_robot_combinations(self):
        """Gives possible robot combinations for constraints
        construction."""
        robot_combinations = combinations(range(self.swarm.specs.n_robot),2)
        return list(robot_combinations)
    
    def __set_mode_sequence(self):
        """Gives the current mode sequence."""
        current_mode = self.swarm.mode
        mode_sequence = deque(range(1,self.swarm.specs.n_mode))
        mode_sequence.rotate(-current_mode+1)
        mode_sequence.appendleft(0)
        return list(mode_sequence)
    
    def __construct_vars(self):
        """Constructing casadi symbolic variables for optimization."""
        n_inner = self.n_inner
        n_outer = self.n_outer
        n_mode = self.swarm.specs.n_mode
        n_robot = self.swarm.specs.n_robot
        X = ca.SX.sym('x',2*n_robot,n_outer*(n_inner+1)*(n_mode-1))
        U = ca.SX.sym('u',2,n_outer*(n_inner+1)*(n_mode-1))
        P = ca.SX.sym('p',2*n_robot,2)
        counter = 0
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                for i_inner in range(n_inner+1):
                    varstr ='{:02d}_{:02d}_{:02d}'.format(i_outer,mode,i_inner)
                    for robot in range(n_robot):
                        rob_str = '_{:02d}'.format(robot)
                        X[2*robot,counter] = ca.SX.sym('x_'+varstr+rob_str)
                        X[2*robot + 1,counter] = ca.SX.sym('y_'+varstr+rob_str)
                    if i_inner == n_inner:
                    # if this is last steps of this mode use rotation
                        i_inner = 0
                        mode = 0
                        varstr ='{:02d}_{:02d}_{:02d}'.format(i_outer,
                                                              mode,i_inner)
                    U[0,counter] = ca.SX.sym('r_'+varstr+rob_str)
                    U[1,counter] = ca.SX.sym('t_'+varstr+rob_str)
                    counter += 1
        for robot in range(n_robot):
            P[2*robot,0] = ca.SX.sym('xi_{:02d}'.format(robot))
            P[2*robot + 1,0] = ca.SX.sym('yi_{:02d}'.format(robot))
            P[2*robot,1] = ca.SX.sym('xf_{:02d}'.format(robot))
            P[2*robot + 1,1] = ca.SX.sym('yf_{:02d}'.format(robot))
        return X, U, P

    def set_constraint(self):
        pass

########## test section ################################################
if __name__ == '__main__':
    swarm_specs = model.SwarmSpecs(np.array([[9,7,5,3],[3,5,7,9],[3,2,6,9]]),
                                   5, 10)
    swarm = model.Swarm(np.array([0,0,10,0,20,0,30,0]), 0, 1, swarm_specs)
    planner = Planner(swarm)

    #print(planner.swarm.position)
    #print(planner.swarm.specs.B)
    #print(planner.swarm.specs.beta)
    #print(planner.robot_combinations)
    #print(planner.mode_sequence)
    print(planner.X.T)
    print(planner.U.T)
    print(planner.P.T)
"""     print(planner.swarm.specs.n_robot)
    print(planner.swarm.specs.n_mode)
    print(planner.n_inner)
    print(planner.n_outer) """