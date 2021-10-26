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
        self.ub_space_x = np.inf #105
        self.lb_space_x = -np.inf #-105
        self.ub_space_y = np.inf #85
        self.lb_space_y = -np.inf #-85
        self.X, self.U, self.P = self.__construct_vars()
        self.lbg, self.ubg = [None]*2
        self.lbx, self.ubx = [None]*2
        self.solver = None
        self.xi = self.swarm.position
        self.xf = None
    
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
        X = ca.SX.sym('x',2*n_robot,n_outer*n_inner*(n_mode-1))
        U = ca.SX.sym('u',2,n_outer*(n_inner*(n_mode-1) + 1))
        P = ca.SX.sym('p',2*n_robot,2)
        counter = 0
        counter_u = 0
        # Building X and U
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                # Maps to current mode sequence.
                mode = mode_sequence[mode]
                for i_inner in range(n_inner):
                    varstr ='{:02d}_{:02d}_{:02d}'.format(i_outer,mode,i_inner)
                    for robot in range(n_robot):
                        rob_str = '_{:02d}'.format(robot)
                        X[2*robot,counter] = ca.SX.sym('x_'+varstr+rob_str)
                        X[2*robot + 1,counter] = ca.SX.sym('y_'+varstr+rob_str)
                    
                    U[0,counter_u] = ca.SX.sym('dx_'+varstr)
                    U[1,counter_u] = ca.SX.sym('dy_'+varstr)
                    counter += 1
                    counter_u += 1
            varstr = '{:02d}_{:02d}_00'.format(i_outer,0)
            U[0,counter_u] = ca.SX.sym('dx_'+varstr)
            U[1,counter_u] = ca.SX.sym('dy_'+varstr)
            counter_u += 1
        # Building P
        for robot in range(n_robot):
            P[2*robot,0] = ca.SX.sym('xi_{:02d}'.format(robot))
            P[2*robot + 1,0] = ca.SX.sym('yi_{:02d}'.format(robot))
            P[2*robot,1] = ca.SX.sym('xf_{:02d}'.format(robot))
            P[2*robot + 1,1] = ca.SX.sym('yf_{:02d}'.format(robot))

        return X, U, P
    
    def __f(self,state,control,mode):
        """Defines swarm transition for CASADI."""
        B = self.swarm.specs.B
        next_state = state + ca.mtimes(B[mode,:,:],control)
        return next_state

    def get_constraint_inter_robot(self,x,u,mode):
        """Returns inter-robot constraints.
        
        It uses u = [r, theta] as input."""
        g = []
        dm = self.d_min
        beta = self.swarm.specs.beta[mode,:]
        dx = u[0]
        dy = u[1]
        for pair in self.robot_pairs:
            zi = x[2*pair[0]:2*pair[0]+2]
            zj = x[2*pair[1]:2*pair[1]+2]
            a = zi[0]-zj[0]
            b = zi[1] - zj[1]
            bij = beta[pair[0]] - beta[pair[1]]
            g += [bij*(a*dx+b*dy)
                  +ca.fabs(bij*ca.sqrt(a**2+b**2-dm**2)*(-b*dx+a*dy)/dm)]
        return g
    
    def get_constraint_shooting(self,x_next, x, u, mode):
        """Returns constraint resulted from multiple shooting."""
        B = self.swarm.specs.B[mode,:,:]
        g = []
        g += [x_next - x - ca.mtimes(B,u)]
        return g

    def get_constraints(self):
        """This function builds constraints of optimization."""
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
        counter_u = 0
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                # Maps to mode sequence
                mode_mapped = mode_sequence[mode]
                for i_inner in range(n_inner):
                    st = X[:,counter]
                    control = U[:,counter_u]
                    st_next = X[:,counter+1]
                    g_shooting += self.get_constraint_shooting(st_next, st,
                                                               control,
                                                               mode_mapped)
                    g_inter_robot += self.get_constraint_inter_robot(st_next,
                                                                     control)
                    counter += 1
                    counter_u += 1
                if mode < n_mode -1:
                    # If this is not last mode of current outer loop,
                    # then take one rotation in the last input direction.
                    # else do nothing and proceed to outer loop transition.
                    mode = 0
                    st = X[:,counter]
                    control[0] = rotation_distance
                    st_next = X[:,counter+1]
                    g_shooting += self.get_constraint_shooting(st_next, st,
                                                               control, mode)
                    counter += 1

            if (i_outer < n_outer -1):
                # If this is not the last outer loop
                # Take multiple of n_mode-1 rotations plus one rotation
                mode = 0
                st = X[:,counter]
                control = U[:,counter_u]
                control[0] = ((n_mode-1)*rotation_distance*control[0]
                              + rotation_distance)
                st_next = X[:,counter+1]
                g_shooting += self.get_constraint_shooting(st_next, st,
                                                           control, mode)
                counter += 1
                counter_u += 1
        # Take last step
        st = X[:,counter]
        control = U[:,counter_u]
        control[0] = rotation_distance*control[0]
        st_next = P[:,1]
        g_shooting += self.get_constraint_shooting(st_next, st,
                                                   control, mode)
        g = ca.vertcat(*(g_shooting + g_inter_robot))
        # upper bound on g
        n_g_shooting = ca.vertcat(*g_shooting).shape[0]
        n_g_inter_robot = ca.vertcat(*g_inter_robot).shape[0]
        lbg = np.hstack((np.zeros(n_g_shooting),
                         np.zeros(n_g_inter_robot) ))
        ubg = np.hstack((np.zeros(n_g_shooting),
                         np.inf*np.ones(n_g_inter_robot) ))
        return g, lbg, ubg

    def get_optim_vars(self):
        """This function returns optimization flatten variable, its
        bounds, and the discrete vector that indicates integer
        variables."""
        U = self.U
        X = self.X
        P  = self.P
        n_inner = self.n_inner
        n_outer = self.n_outer
        n_robot = self.swarm.specs.n_robot
        n_mode = self.swarm.specs.n_mode
        optim_var = ca.vertcat(ca.reshape(U,-1,1), ca.reshape(X,-1,1))
        # CASADI vert cat does column wise operation.
        # Bounds related to U
        discrete_U = []
        lbu = []
        ubu = []
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                for i_inner in range(n_inner):
                    discrete_U += [False]*2
                    lbu += [0,0]
                    ubu += [np.inf,2*np.pi]
            discrete_U += [True,False]
            lbu += [0,0]
            ubu += [np.inf,2*np.pi]
        # Bounds related to X
        discrete_X = []
        lbxx = []
        ubxx =[]
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                for i_inner in range(n_inner+1):
                    discrete_X += [False]*2*n_robot
                    lbxx += [self.lb_space_x, self.lb_space_y]*n_robot
                    ubxx += [self.ub_space_x, self.ub_space_y]*n_robot
        # concatenating X and U bounds
        discrete = discrete_U + discrete_X
        lbx = np.array(lbu + lbxx)
        ubx = np.array(ubu + ubxx)
        # concatenating optimization parameter
        p = ca.reshape(P, -1, 1)
        return optim_var, lbx, ubx, discrete, p

    def get_objective(self, sparse = False):
        """Returns objective function for optimization.
        If sparse = True, then it returns first norm objective function
        that favors sparsity.
        """
        r = self.U[0,:].T
        obj = 0
        if sparse is False:
            obj += ca.sum1(r*r)
        else:
            obj = ca.norm_1(r)
        return obj

    def get_optimization(self, is_discrete = False, is_sparse = False):
        """Sets up and returns a CASADI optimization object."""
        g, lbg, ubg = self.get_constraints()
        optim_var, lbx, ubx, discrete, p = self.get_optim_vars()
        obj = self.get_objective(is_sparse)
        nlp_prob = {'f': obj, 'x': optim_var, 'g': g, 'p': p}
        if is_discrete is False:
            # Use ipopt solver and consider all variables as continuous.
            opts = {}
            #opts['ipopt.print_level'] = 0
            #opts['print_time'] = 0
            solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        else:
            opts = {}
        self.lbg, self.ubg = lbg, ubg
        self.lbx, self.ubx = lbx, ubx
        self.discrete = discrete
        self.solver = solver
        return solver
    def __post_process_u(self,sol):
        """Post processes the solution and adds intermediate steps."""
        mode_sequence = self.mode_sequence
        rotation_distance = self.swarm.specs.rotation_distance
        n_inner = self.n_inner
        n_outer = self.n_outer
        n_robot = self.swarm.specs.n_robot
        n_mode = self.swarm.specs.n_mode
        U_sol = sol['x'][:2*n_outer*((n_inner)*(n_mode-1) + 1)]
        X_sol = sol['x'][2*n_outer*((n_inner)*(n_mode-1) + 1):]
        U_sol = ca.reshape(U_sol,2,-1).full()
        X_sol = ca.reshape(X_sol,2*n_robot,-1).full()
        U = np.zeros((3,n_outer*(n_inner+1)*(n_mode-1)))
        counter = 0
        counter_u = 0
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                mode_mapped = mode_sequence[mode]
                for i_inner in range(n_inner):
                    U[:2,counter_u] = U_sol[:,counter]
                    U[2,counter_u] = mode_mapped
                    counter += 1
                    counter_u += 1
                if mode < n_mode -1:
                    # if this is not the last iteration of this outer loop
                    U[0,counter_u] = rotation_distance
                    U[1,counter_u] = U_sol[1,counter -1]
                    U[2,counter_u] = 0
                    counter_u +=1
                else:
                    # if this is last iteration of current outer loop
                    if i_outer< n_outer -1:
                        # if this is not last outer loop
                        U[0,counter_u] = ((n_mode - 1)
                                           *rotation_distance
                                           *U_sol[0,counter]
                                          + rotation_distance)
                        U[1,counter_u] = U_sol[1,counter]
                        U[2,counter_u] = 0
                        counter += 1
                        counter_u += 1
                    else:
                        # if this is last outer loop
                        U[0,counter_u] = rotation_distance*U_sol[0,counter]
                        U[1,counter_u] = U_sol[1,counter]
                        U[2,counter_u] = 0
                        pass
        return U_sol, X_sol, U


    def solve_optimization(self, xf, is_discrete=False, is_sparse=False):
        """Solves the optimization problem, sorts and post processes the
        answer and returns the answer.
        """
        lbx, ubx = self.lbx, self.ubx
        lbg, ubg = self.lbg, self.ubg
        solver = self.solver
        xi = self.xi

        U0 = ca.DM.zeros(self.U.shape)
        X0 = ca.DM.zeros(self.X.shape)
        x0 = ca.vertcat(ca.reshape(U0,-1,1), ca.reshape(X0,-1,1))
        p = np.hstack((xi, xf))
        sol = solver(x0 = x0, lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg, p = p)
        # recovering the solution in appropriate format
        U_sol, X_sol, U = self.__post_process_u(sol)
        return sol, U_sol, X_sol, U
        




########## test section ################################################
if __name__ == '__main__':
    swarm_specs = model.SwarmSpecs(np.array([[10,5,3],[3,5,10]]),
                                   5, 10)
    swarm = model.Swarm(np.array([0,0,10,0,20,0]), 0, 1, swarm_specs)
    planner = Planner(swarm, n_inner = 2, n_outer = 2)

    #print(planner.swarm.position)
    #print(planner.swarm.specs.B)
    #print(planner.swarm.specs.beta)
    #print(planner.robot_combinations)
    #print(planner.mode_sequence)
    #print(planner.build_optimization())
    #print(planner.X.T)
    #print(planner.U.T)
    #print(planner.P.T)
    #g, lbg, ubg = planner.get_constraints()
    #optim_var, lbx, ubx, discrete, p = planner.get_optim_vars()
    #obj = planner.get_objective(sparse = False)
    #nlp_prob = {'f': obj, 'x': optim_var, 'g': g, 'p': p}
    #solver = planner.get_optimization()
    #xf = np.array([0,40,10,40,20,40])
    #sol, U_sol, X_sol, U = planner.solve_optimization(xf)

    #anim = swarm.simanimation(U,1000)
    #x = ca.SX.sym('x',4*2)
    #u = ca.SX.sym('u',2)
    #i = 2
    #print(planner.swarm.specs.B[i,:,:])
    #print(planner.f(x,u,i))
    #print(planner.swarm.specs.n_robot)
    #print(planner.swarm.specs.n_mode)
    #print(planner.n_inner)
    #print(planner.n_outer)