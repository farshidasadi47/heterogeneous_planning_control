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
        self.robot_pairs = self.__set_robot_pairs()
        self.mode_sequence = self.__set_mode_sequence()
        self.d_min = 10
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
    
    def __set_robot_pairs(self):
        """Gives possible unique robot pairs for constraints
        construction."""
        robot_pairs = combinations(range(self.swarm.specs.n_robot),2)
        return list(robot_pairs)
    
    def __set_mode_sequence(self):
        """Gives the current mode sequence."""
        current_mode = self.swarm.mode
        mode_sequence = deque(range(1,self.swarm.specs.n_mode))
        mode_sequence.rotate(-current_mode+1)
        mode_sequence.appendleft(0)
        return list(mode_sequence)
    
    def __construct_vars(self):
        """Constructing casadi symbolic variables for optimization.
        
        """
        n_inner = self.n_inner
        n_outer = self.n_outer
        n_mode = self.swarm.specs.n_mode
        n_robot = self.swarm.specs.n_robot
        mode_sequence = self.mode_sequence
        X = ca.SX.sym('x',2*n_robot,n_outer*(n_inner+1)*(n_mode-1))
        U = ca.SX.sym('u',2,n_outer*((n_inner)*(n_mode-1) + 1))
        P = ca.SX.sym('p',2*n_robot,2)
        counter = 0
        counter_u = 0
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                # Maps to current mode sequence.
                mode = mode_sequence[mode]
                for i_inner in range(n_inner+1):
                    varstr ='{:02d}_{:02d}_{:02d}'.format(i_outer,mode,i_inner)
                    for robot in range(n_robot):
                        rob_str = '_{:02d}'.format(robot)
                        X[2*robot,counter] = ca.SX.sym('x_'+varstr+rob_str)
                        X[2*robot + 1,counter] = ca.SX.sym('y_'+varstr+rob_str)
                    if i_inner == n_inner:
                    # if this is last steps of this mode, rotate for one
                    # step and only consider resulted X
                        counter += 1
                        break
                    U[0,counter_u] = ca.SX.sym('r_'+varstr)
                    U[1,counter_u] = ca.SX.sym('t_'+varstr)
                    counter += 1
                    counter_u += 1
            varstr = '{:02d}_{:02d}_00'.format(i_outer,0)
            U[0,counter_u] = ca.SX.sym('r_'+varstr)
            U[1,counter_u] = ca.SX.sym('t_'+varstr)
            counter_u += 1

        for robot in range(n_robot):
            P[2*robot,0] = ca.SX.sym('xi_{:02d}'.format(robot))
            P[2*robot + 1,0] = ca.SX.sym('yi_{:02d}'.format(robot))
            P[2*robot,1] = ca.SX.sym('xf_{:02d}'.format(robot))
            P[2*robot + 1,1] = ca.SX.sym('yf_{:02d}'.format(robot))
        return X, U, P
    
    def f(self,state,control,mode):
        """Defines swarm transition for CASADI."""
        B = self.swarm.specs.B
        next_state = state + ca.mtimes(B[mode,:,:],control)
        return next_state

    def get_constraint_inter_robot(self,x,u):
        """Returns inter-robot constraints.
        
        It uses u = [r, theta] as input."""
        g = []
        dm = self.d_min
        uh = ca.vertcat(-ca.sin(u[1]), ca.cos(u[1]))
        for pair in self.robot_pairs:
            zi = x[2*pair[0]:2*pair[0]+2]
            zj = x[2*pair[1]:2*pair[1]+2]
            g += [ca.dot(zi-zj,zi-zj) - ca.dot(zi-zj,uh) - dm**2]
        return g
    
    def get_constraint_shooting(self,x_next, x, u, mode):
        """Returns constraint resulted from multiple shooting."""
        B = self.swarm.specs.B[mode,:,:]
        ur = ca.vertcat(-u[0]*ca.sin(u[1]), u[0]*ca.cos(u[1]))
        g = []
        g += [x_next - x - ca.mtimes(B,ur)]
        return g

    def build_optimization(self):
        """This function builds optimization objective, constraints,
        and var bounds."""
        mode_sequence = self.mode_sequence
        n_mode = self.swarm.specs.n_mode
        n_robot = self.swarm.specs.n_robot
        n_inner = self.n_inner
        n_outer = self.n_outer
        X = self.X
        U = self.U
        P = self.P
        d_min  = self.d_min
        rotation_distance = self.swarm.specs.rotation_distance
        obj = 0
        g_shooting = []
        g_shooting += [P[:,0] - X[:,0]]
        g_inter_robot = []
        counter = 0
        for i_outer in range(n_outer):
            if i_outer == n_outer:
                n_inner = n_inner -1
            for mode in range(1,n_mode):
                mode = mode_sequence[mode]
                for i_inner in range(n_inner):
                    st = X[:,counter]
                    st_next = X[:,counter+1]
                    control = U[:,counter]
                    g_shooting += self.get_constraint_shooting(st_next, st,
                                                               control, mode)
                    counter += 1
                mode = 0

        return 0

########## test section ################################################
if __name__ == '__main__':
    swarm_specs = model.SwarmSpecs(np.array([[9,7,5,3],[3,5,7,9],[3,2,6,9]]),
                                   5, 10)
    swarm = model.Swarm(np.array([0,0,10,0,20,0,30,0]), 0, 1, swarm_specs)
    swarm.mode = 3
    planner = Planner(swarm)

    #print(planner.swarm.position)
    #print(planner.swarm.specs.B)
    #print(planner.swarm.specs.beta)
    #print(planner.robot_combinations)
    #print(planner.mode_sequence)
    #print(planner.build_optimization())
    print(planner.X.T)
    print(planner.U.T)
    print(planner.P.T)
    
    #x = ca.SX.sym('x',4*2)
    #u = ca.SX.sym('u',2)
    #i = 2
    #print(planner.swarm.specs.B[i,:,:])
    #print(planner.f(x,u,i))
    print(planner.swarm.specs.n_robot)
    print(planner.swarm.specs.n_mode)
    print(planner.n_inner)
    print(planner.n_outer)