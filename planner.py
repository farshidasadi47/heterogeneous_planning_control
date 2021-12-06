########################################################################
# This files hold classes and functions that plans swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from itertools import combinations
from collections import deque

import numpy as np
import numpy.matlib
import casadi as ca

import model

np.set_printoptions(precision=2, suppress=True)
########## classes and functions #######################################
class Planner():
    """This class contains objects and methods for planning a path
    for a given swarm of milirobots."""

    def __init__(self, swarm: model.Swarm, n_outer = 3):
        self.swarm = swarm
        self.robot_pairs = self.__set_robot_pairs()
        self.mode_sequence = self.__set_mode_sequence()
        self.d_min = swarm.specs.tumbling_distance
        self.x_final = None
        self.n_outer = n_outer
        self.ub_space_x = None
        self.lb_space_x = None
        self.ub_space_y = None
        self.lb_space_y = None
        self.ub_wrap_x = None
        self.lb_wrap_x = None
        self.ub_wrap_y = None
        self.lb_wrap_y = None
        # Setting the feasible limits
        self.set_space_limit(swarm.specs.ubx, swarm.specs.lbx,
                             swarm.specs.uby, swarm.specs.lby)
        self.X, self.U, self.P = self.__construct_vars()
        self.lbg, self.ubg = [None]*2
        self.lbx, self.ubx = [None]*2
        self.solver_opt = self.__solvers()
        self.solver = None
        self.xi = self.swarm.position
        self.xf = None
    
    def set_space_limit(self, ubx, lbx, uby, lby):
        """Sets space boundary limits and updates the bands in
        optimization problem."""
        # Setting feasible space for optimization.
        self.ub_space_x = ubx - self.swarm.specs.tumbling_distance*2.5
        self.lb_space_x = lbx + self.swarm.specs.tumbling_distance*2.5
        self.ub_space_y = uby - self.swarm.specs.tumbling_distance*2.5
        self.lb_space_y = lby + self.swarm.specs.tumbling_distance*2.5
        # Set feasible space for post process wrapping.
        self.ub_wrap_x = ubx-self.swarm.specs.tumbling_distance/2
        self.lb_wrap_x = lbx+self.swarm.specs.tumbling_distance/2
        self.ub_wrap_y = uby-self.swarm.specs.tumbling_distance/2
        self.lb_wrap_y = lby+self.swarm.specs.tumbling_distance/2
    
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
        n_outer = self.n_outer
        n_mode = self.swarm.specs.n_mode
        n_robot = self.swarm.specs.n_robot
        mode_sequence = self.mode_sequence
        X = ca.SX.sym('x',2*n_robot,n_outer*(n_mode-1) + 1)
        U = ca.SX.sym('u',2,n_outer*(n_mode-1)+1)
        P = ca.SX.sym('p',2*n_robot,2)
        counter = 0
        counter_u = 0
        # Build X and U.
        # Construct position in the start of outer loop.
        i_outer = 0
        mode = 0
        varstr ='{:02d}_{:02d}'.format(i_outer,mode)
        for robot in range(n_robot):
            rob_str = '_{:02d}'.format(robot)
            X[2*robot,counter] = ca.SX.sym('x_'+varstr+rob_str)
            X[2*robot + 1,counter] = ca.SX.sym('y_'+varstr+rob_str)
        counter += 1
        for i_outer in range(n_outer):
            # Construct position and control for current outer loop.
            for mode in range(1,n_mode):
                # Maps to current mode sequence.
                mode = mode_sequence[mode]
                varstr ='{:02d}_{:02d}'.format(i_outer,mode)
                for robot in range(n_robot):
                    rob_str = '_{:02d}'.format(robot)
                    X[2*robot,counter] = ca.SX.sym('x_'+varstr+rob_str)
                    X[2*robot + 1,counter] = ca.SX.sym('y_'+varstr+rob_str)
                    
                U[0,counter_u] = ca.SX.sym('dx_'+varstr)
                U[1,counter_u] = ca.SX.sym('dy_'+varstr)
                counter += 1
                counter_u += 1
        # Construct control to go for next outer loop.
        varstr = '{:02d}_{:02d}'.format(i_outer,0)
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
    
    def f(self,state,control,mode):
        """Defines swarm transition for CASADI."""
        B = self.swarm.specs.B
        next_state = state + ca.mtimes(B[int(mode),:,:],control)
        return next_state.full().squeeze()

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
                  +ca.fabs(bij*ca.sqrt(ca.fabs(a**2+b**2-dm**2))
                   *(-b*dx+a*dy)/dm),
                  a**2+b**2-dm**2]
        return g
    
    def get_constraint_shooting(self,x_next, x, u, mode):
        """Returns constraint resulted from multiple shooting."""
        B = self.swarm.specs.B[mode,:,:]
        g = []
        g += [x_next - x - ca.mtimes(B,u)]
        return g
    
    def get_constraint_distance(self,x_next):
        """"This function return the constraint with respective to
        maximum distance between each milirobot for each axis."""
        g = []
        for pair in self.robot_pairs:
            zi = x_next[2*pair[0]:2*pair[0]+2]
            zj = x_next[2*pair[1]:2*pair[1]+2]
            g += [zi-zj]
        return g

    def get_constraints(self, boundary=False):
        """This function builds constraints of optimization."""
        mode_sequence = self.mode_sequence
        n_mode = self.swarm.specs.n_mode
        n_outer = self.n_outer
        X = self.X
        U = self.U
        P = self.P
        g_shooting = []
        g_shooting += [P[:,0] - X[:,0]]
        g_inter_robot = []
        g_distance = []
        counter = 0
        counter_u = 0
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                # Maps to mode sequence
                mode_mapped = mode_sequence[mode]
                st = X[:,counter]
                control = U[:,counter_u]
                st_next = X[:,counter+1]
                g_shooting += self.get_constraint_shooting(st_next, st,
                                                           control,
                                                           mode_mapped)
                g_inter_robot += self.get_constraint_inter_robot(st,
                                                                 control,
                                                                 mode_mapped)
                g_distance += self.get_constraint_distance(st_next)
                counter += 1
                counter_u += 1
        # Take last step.
        mode = 0
        st = X[:,counter]
        control = U[:,counter_u]
        st_next = P[:,1]
        g_shooting += self.get_constraint_shooting(st_next, st,
                                                   control, mode)
        # Configure bounds of g
        n_g_shooting = ca.vertcat(*g_shooting).shape[0]
        n_g_inter_robot = ca.vertcat(*g_inter_robot).shape[0]
        n_g_distance = ca.vertcat(*g_distance).shape[0]
        
        lbdsx = [-(self.ub_space_x-self.lb_space_x),
               -(self.ub_space_y-self.lb_space_y)]*(n_g_distance//2)
        ubdsx = [self.ub_space_x-self.lb_space_x,
               self.ub_space_y-self.lb_space_y]*(n_g_distance//2)
        if boundary ==True:
            g = ca.vertcat(*(g_shooting + g_inter_robot + g_distance))
            lbg = np.hstack((np.zeros(n_g_shooting),
                             np.zeros(n_g_inter_robot),
                             np.array(lbdsx) ))
            ubg = np.hstack((np.zeros(n_g_shooting),
                             np.inf*np.ones(n_g_inter_robot),
                             np.array(ubdsx) ))
        else:
            g = ca.vertcat(*(g_shooting + g_inter_robot))
            lbg = np.hstack((np.zeros(n_g_shooting),
                             np.zeros(n_g_inter_robot) ))
            ubg = np.hstack((np.zeros(n_g_shooting),
                             np.inf*np.ones(n_g_inter_robot) ))
        return g, lbg, ubg

    def get_optim_vars(self):
        """This function returns optimization flatten variable and its
        bounds."""
        U = self.U
        X = self.X
        P  = self.P
        n_outer = self.n_outer
        n_robot = self.swarm.specs.n_robot
        n_mode = self.swarm.specs.n_mode
        optim_var = ca.vertcat(ca.reshape(U,-1,1), ca.reshape(X,-1,1))
        # CASADI vert cat does column wise operation.
        # Configure bounds for U.
        lbu = []
        ubu = []
        lbuu = [-np.inf,-np.inf]
        ubuu = [np.inf,np.inf]
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                lbu += lbuu
                ubu += ubuu
        # Configure bounds for last step in current outer loop.
        lbu += [-np.inf,-np.inf]
        ubu += [np.inf,np.inf]

        # Configure bounds related to X.
        lbxx = [-np.inf]*2*n_robot*(n_outer*(n_mode-1) + 1)
        ubxx = [np.inf]*2*n_robot*(n_outer*(n_mode-1) + 1)
        # concatenating X and U bounds
        lbx = np.array(lbu + lbxx)
        ubx = np.array(ubu + ubxx)
        # concatenating optimization parameter
        p = ca.reshape(P, -1, 1)
        return optim_var, lbx, ubx, p

    def get_objective(self):
        """Returns objective function for optimization.
        If sparse = True, then it returns first norm objective function
        that favors sparsity.
        """
        U = self.U
        obj = 0
        for i in range(U.shape[1]):
            u = U[:,i]
            obj += ca.sum1(u*u)

        return obj

    def get_optimization(self, solver_name = 'ipopt', boundary = False):
        """Sets up and returns a CASADI optimization object."""
        g, lbg, ubg = self.get_constraints(boundary)
        optim_var, lbx, ubx, p = self.get_optim_vars()
        obj = self.get_objective()
        nlp_prob = {'f': obj, 'x': optim_var, 'g': g, 'p': p}
        _solver = self.solver_opt[solver_name]

        solver = ca.nlpsol('solver', _solver['name'],
                           nlp_prob, _solver['opts'])
        
        self.lbg, self.ubg = lbg, ubg
        self.lbx, self.ubx = lbx, ubx
        self.solver = solver
        return solver
    def __solvers(self):
        """This function build a dictionary of different solvers and
        their corresponding options.
        """
        solvers = {}
        # Configuring different options for solverws
        solvers['ipopt'] = {}
        solvers['ipopt']['name'] = 'ipopt'
        solvers['ipopt']['opts'] = {}

        # See knitro online manual for more information
        solvers['knitro'] = {}
        solvers['knitro']['name'] = 'knitro'
        solvers['knitro']['opts'] = {}
        
        # max time for each series of iterations in seconds
        solvers['knitro']['opts']['knitro.maxtime_real'] = 3
        # non convex
        solvers['knitro']['opts']['knitro.convex'] = 0
        # multiple start, ends when finds first feasible solution
        solvers['knitro']['opts']['knitro.ms_enable'] = 1
        solvers['knitro']['opts']['knitro.ms_terminate'] = 2
        # emphasize feasibility
        solvers['knitro']['opts']['knitro.bar_feasible'] = 2
        solvers['knitro']['opts']['knitro.bar_switchrule'] = 3
        # how many fine iteration to do after interior point method
        #solvers['knitro']['opts']['knitro.bar_maxcrossit'] = 30

        return solvers

    def __cartesian_to_polar(self,z):
        """Converts cartesian to polar coordinate based on Y coordinate
        as reference for zero angle."""
        z = complex(z[0],z[1])
        z = np.array([np.abs(z), np.angle(z)])
        if z[0] == 0:
            z[1] = 0
        return z

    def __accurate_rotation(self,u):
        """This function gets a desired rotation and returns a sequence
        of two pure steps of rotations that produce the desired movement
        in rotation mode."""
        rotation_distance = self.swarm.specs.rotation_distance
        r = np.linalg.norm(u)
        if r>0:
            r1 = max(np.floor(r/rotation_distance)*rotation_distance,
                     rotation_distance)
            if r%rotation_distance > 0:
                r2 = rotation_distance
                teta = ca.SX.sym('t',2)
                f1 = r1*ca.cos(teta[0])+r2*ca.cos(teta[1]) - u[0]
                f2 = r1*ca.sin(teta[0])+r2*ca.sin(teta[1]) - u[1]
                f = ca.Function('g',[teta],[ca.vertcat(*[f1,f2])])
                F = ca.rootfinder('F','newton',f)
                teta_value = F(np.random.rand(2))
            else:
                r2 = 0
                teta_value = np.zeros(2)
                teta_value[0] = self.__cartesian_to_polar(u)[1]
            u_possible = np.zeros((2,2))
            u_possible[0,0] = r1*np.cos(teta_value[0])
            u_possible[1,0] = r1*np.sin(teta_value[0])
            u_possible[0,1] = r2*np.cos(teta_value[1])
            u_possible[1,1] = r2*np.sin(teta_value[1])
        else:
            u_possible = np.zeros((2,2))
        return u_possible

    def __post_process_u(self,sol):
        """Post processes the solution and adds intermediate steps."""
        mode_sequence = self.mode_sequence
        rotation_distance = self.swarm.specs.rotation_distance
        n_outer = self.n_outer
        n_robot = self.swarm.specs.n_robot
        n_mode = self.swarm.specs.n_mode
        xi = self.xi
        U_sol = sol['x'][:2*(n_outer*(n_mode-1) + 1)]
        X_sol = sol['x'][2*(n_outer*(n_mode-1) + 1):]
        U_sol = ca.reshape(U_sol,2,-1).full()
        X_sol = ca.reshape(X_sol,2*n_robot,-1).full()
        UZ = np.zeros((3,1+n_outer*2*(n_mode-1)))
        X = np.zeros((2*n_robot, 1+n_outer*2*(n_mode-1) ))
        X[:,0] = xi
        U = np.zeros_like(UZ)
        # recovering input with transitions
        counter = 0
        counter_u = 0
        change_direction = 1
        rotation_remainder = np.zeros(2)
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                # map the mode to the current swarm mode sequence.
                mode_mapped = mode_sequence[mode]
                UZ[:2,counter_u] = U_sol[:,counter]
                UZ[2,counter_u] = mode_mapped
                X[:,counter_u + 1] = self.f(X[:,counter_u],
                                              UZ[:2,counter_u],
                                              UZ[2,counter_u])
                counter += 1
                counter_u +=1
                
                # Change mode
                mode = 0
                UZ[0,counter_u] = 0
                UZ[1,counter_u] = rotation_distance*change_direction
                UZ[2,counter_u] = mode
                X[:,counter_u + 1] = self.f(X[:,counter_u],
                                              UZ[:2,counter_u],
                                              UZ[2,counter_u])
                # Keep track of rotations done.
                change_direction *= -1
                rotation_remainder += UZ[:2,counter_u]
                counter_u += 1
        rotation_remainder -= UZ[:2,-2]  # Modify mode change done.
        # Refine the last step into two acceptable rotations
        mode = 0
        u_last = U_sol[:,counter] - rotation_remainder
        UZ[:2,-2:] = self.__accurate_rotation(u_last)
        UZ[2,-2:] = np.array([mode,mode])
        X[:,-2] = self.f(X[:,-3],UZ[:2,-2],UZ[2,-2])
        X[:,-1] = self.f(X[:,-2],UZ[:2,-1],UZ[2,-1])
        # Calculate the corresponding polar coordinate inputs.
        U[2,:] = UZ[2,:]
        for i in range(UZ.shape[1]):
            U[:2,i] = self.__cartesian_to_polar(UZ[:2,i])
            if U[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                U[0,i] = round(U[0,i])
        
        return U_sol, X_sol, UZ,  U, X

    def __post_process_wrap(self,sol):
        """Post processes the solution.
        1- Adds intermediate steps to change mode.
        2- Wraps inputs so that the swarm stays inside the limits."""
        mode_sequence = self.mode_sequence
        n_outer = self.n_outer
        n_robot = self.swarm.specs.n_robot
        n_mode = self.swarm.specs.n_mode
        xi = self.xi
        U_sol = sol['x'][:2*(n_outer*(n_mode-1) + 1)]
        X_sol = sol['x'][2*(n_outer*(n_mode-1) + 1):]
        U_sol = ca.reshape(U_sol,2,-1).full()
        X_sol = ca.reshape(X_sol,2*n_robot,-1).full()
        X = []
        UZ = []
        X.append(xi)
        x = X[-1]
        counter = 0
        rotation_done = np.zeros(2)
        uhat_old = np.array([0,1.0])
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                # map the mode to the current swarm mode sequence.
                mode_mapped = mode_sequence[mode]
                wrap=self.__wrap_u(U_sol[:,counter],x,mode_mapped,uhat_old)
                X_wrap, UZ_wrap, wrap_done, uhat_old = wrap
                X.extend(X_wrap)
                UZ.extend(UZ_wrap)
                x = X[-1]
                rotation_done += wrap_done
                counter += 1
        # Take last step rotation
        # Remove the last mode change
        rotation_done = rotation_done - UZ[-1][:2]
        X.pop()
        UZ.pop()
        x = X[-1]
        # Calculate the last rotation
        # Refine the last step into two acceptable rotations
        mode = 0
        u_last = np.zeros((3,2))
        rotation_remainder = U_sol[:,counter] - rotation_done
        u_last[:2,:] = self.__accurate_rotation(rotation_remainder)
        u_last[2,-2:] = np.array([mode,mode])
        # Update and record values
        x = self.f(x,u_last[:2,0],mode)
        X.append(x)
        UZ.append(u_last[:,0])
        x = self.f(x,u_last[:2,1],mode)
        X.append(x)
        UZ.append(u_last[:,1])
        #
        X = np.array(X).T
        UZ = np.array(UZ).T
        # Calculate the corresponding polar coordinate inputs.
        U = np.zeros_like(UZ)
        U[2,:] = UZ[2,:]
        for i in range(UZ.shape[1]):
            U[:2,i] = self.__cartesian_to_polar(UZ[:2,i])
            if U[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                U[0,i] = round(U[0,i])

        return U_sol, X_sol, UZ, U, X
    
    def __wrap_u(self,u,x,mode,uhat_old):
        """This function wraps the current section of the input,
        so that the swarm srays in the space limits."""
        X_wrap = []
        UZ_wrap = []
        wrap_done = np.zeros(2)
        rem_u = np.around(u,6)
        rem_r, teta = self.__cartesian_to_polar(rem_u).tolist()
        uhat = np.array([np.cos(teta),np.sin(teta)])
        if rem_r ==0:
            uhat = np.copy(uhat_old)
        while rem_r>0:
            # Determine when the robots reach a wall.
            r_hit = self.__get_max_to_go(x,uhat,mode)
            r_hit = min(rem_r,r_hit)
            # Determine the input
            u_possible = r_hit*uhat
            # Update the remainder of input
            rem_r = rem_r - r_hit
            rem_u = rem_u - u_possible
            # Take the current step.
            x = self.f(x,u_possible,mode)
            # Record the values
            X_wrap.append(x)
            UZ_wrap.append(np.append(u_possible,mode))
            if rem_r == 0:
                break
            # Determine the wrapping if the loop is not broken.
            r_wrap, uhat_wrap = self.__get_roll_to_go(x,uhat,rem_r)
            # Determine the rollings and record it.
            u_possible = r_wrap*uhat_wrap
            wrap_done = wrap_done + u_possible
            # Take the current step.
            x = self.f(x,u_possible,0)  # Zero is rolling mode.
            # Record the values
            X_wrap.append(x)
            UZ_wrap.append(np.append(u_possible,0))  # Zero is rolling mode.
        # Take one roll to change the mode.
        u_possible = self.__mode_change(x,uhat)
        wrap_done = wrap_done + u_possible
        # Take the current step.
        x = self.f(x,u_possible,0)  # Zero is rolling mode.
        # Record the values
        X_wrap.append(x)
        UZ_wrap.append(np.append(u_possible,0))  # Zero is rolling mode.
        return X_wrap, UZ_wrap, wrap_done, uhat

    def __mode_change(self, x, uhat):
        """Changes mode in feasible direction."""
         # Getting the usable space to wrap
        ub_wrap_x = self.ub_wrap_x
        lb_wrap_x = self.lb_wrap_x
        ub_wrap_y = self.ub_wrap_y
        lb_wrap_y = self.lb_wrap_y
        rotation_distance = self.swarm.specs.rotation_distance
        # List of possible preset directions 
        ehat =[uhat,
               np.array([0,1]),np.array([0,-1]),
               np.array([1,0]), np.array([-1,0]),
               np.array([-1,1])/np.sqrt(2),
               np.array([-1,-1])/np.sqrt(2),
               np.array([1,-1])/np.sqrt(2),
               np.array([1,1])/np.sqrt(2)]
        # Check which direction is feasible for mode change.
        for hat in ehat:
            x_next = self.f(x,rotation_distance*hat,0)
            # Get max and min of swarm position in X direction
            xmax = max(x_next[::2]) 
            xmin = min(x_next[::2])
            # Get max and min of swarm position in Y direction
            ymax = max(x_next[1::2]) 
            ymin = min(x_next[1::2])
            # If the swarm is in the feasible space return the input.
            if (xmax<=ub_wrap_x and xmin >=lb_wrap_x and
                ymax<=ub_wrap_y and ymin >=lb_wrap_y):
                u_possible = rotation_distance*hat
                break
        return u_possible

    def __get_max_to_go(self,x,uhat,mode):
        """This function returns the distance_to_go in the current mode
        that the swarm hits a wall in wrap boundary."""
        beta = self.swarm.specs.beta[mode,:]
        n_robot = self.swarm.specs.n_robot
        # Getting the usable space to wrap
        ub_wrap_x = self.ub_wrap_x
        lb_wrap_x = self.lb_wrap_x
        ub_wrap_y = self.ub_wrap_y
        lb_wrap_y = self.lb_wrap_y
        # Get the distance to hit a wall.
        r_hit = np.inf
        for robot in range(n_robot):
            # Get position of current robot.
            zj = x[2*robot:2*robot+2]
            # Check X direction.
            if uhat[0] != 0:
                ubrx = (ub_wrap_x - zj[0])/(beta[robot]*uhat[0])
                lbrx = (lb_wrap_x - zj[0])/(beta[robot]*uhat[0])
            else:
                ubrx = np.inf
                lbrx = -np.inf
            rx = max(ubrx,lbrx)
            # Check Y direction
            if uhat[1] != 0:
                ubry = (ub_wrap_y - zj[1])/(beta[robot]*uhat[1])
                lbry = (lb_wrap_y - zj[1])/(beta[robot]*uhat[1])
            else:
                ubry = np.inf
                ubry = np.inf
            ry = max(ubry,lbry)
            # Calculate when current robot reaches wall.
            rj = min(rx,ry)
            # Update when the swarm reaches to the wall.
            r_hit = min(r_hit,rj)
        return r_hit
    
    def __get_roll_to_go(self,x,uhat,rem_r):
        """This function calculates current wrapping to be done."""
        # Getting the usable space to wrap
        ub_wrap_x = self.ub_wrap_x
        lb_wrap_x = self.lb_wrap_x
        ub_wrap_y = self.ub_wrap_y
        lb_wrap_y = self.lb_wrap_y
        n_robot = self.swarm.specs.n_robot
        rotation_distance = self.swarm.specs.rotation_distance
        n_mode = self.swarm.specs.n_mode
        uhatf = [np.array([-1.0,0]), np.array([1.0,0]),
                 np.array([0,-1.0]), np.array([0,1.0])]
        # Get the direction that is hitting the wall currently.
        ux_clearance = np.inf
        lx_clearance = np.inf
        uy_clearance = np.inf
        ly_clearance = np.inf
        for robot in range(n_robot):
            # Position of current robot.
            zj = x[2*robot:2*robot+2]
            ux_clearance = min(ux_clearance,ub_wrap_x-zj[0])
            lx_clearance = min(lx_clearance,-lb_wrap_x+zj[0])
            uy_clearance = min(uy_clearance,ub_wrap_y-zj[1])
            ly_clearance = min(ly_clearance,-lb_wrap_y+zj[1])
        # Get feasibility guaranteed direction of wrapping.
        clearance = np.array([ux_clearance,lx_clearance,
                              uy_clearance,ly_clearance])
        uhatf = uhatf[np.argmin(clearance)]
        # Get the distance to hit the wall in rolling.
        uhats = [-uhat,uhatf]
        for direction in uhats:
            r_wrap = np.inf
            for robot in range(n_robot):
                # Get position of current robot.
                zj = x[2*robot:2*robot+2]
                # Check X direction.
                if direction[0] != 0:
                    ubrx = (ub_wrap_x - zj[0])/(direction[0])
                    lbrx = (lb_wrap_x - zj[0])/(direction[0])
                else:
                    ubrx = np.inf
                    lbrx = -np.inf
                rx = max(ubrx,lbrx)
                # Check Y direction
                if direction[1] != 0:
                    ubry = (ub_wrap_y - zj[1])/(direction[1])
                    lbry = (lb_wrap_y - zj[1])/(direction[1])
                else:
                    ubry = np.inf
                    ubry = np.inf
                ry = max(ubry,lbry)
                # Calculate when current robot reaches a wall.
                rj = min(rx,ry)
                # Update when the swarm reaches a wall.
                r_wrap = min(r_wrap,rj)
            # Modify max wrapping that maintains the current mode.
            n_wrap = r_wrap//((n_mode-1)*rotation_distance)
            r_wrap = n_wrap*(n_mode-1)*rotation_distance
            # Use the original direction is it gives nonzero wrapping.
            if r_wrap > 0:
                break
        # Check how much wrapping is necessary based on current rem_r.
        rem_n = rem_r//(((n_mode-1)*rotation_distance))
        rem_r = (rem_n+ 1)*(n_mode-1)*rotation_distance
        # Update wrapping needed.
        r_wrap = min(r_wrap,rem_r)
        return r_wrap, direction

    def solve_unconstrained(self,xi,xf):
        """This function solved the unconstrained case from
        controllability analysis and adjusts it for the problem setting,
        to be used as initial guess."""
        mode_sequence = self.mode_sequence
        n_mode = self.swarm.specs.n_mode
        n_robot = self.swarm.specs.n_robot
        n_outer = self.n_outer
        B = np.zeros((2*n_robot,2*n_mode))
        for mode in range(n_mode):
            mapped_mode = mode_sequence[mode]
            B[:,2*mapped_mode:2*mapped_mode +2] = self.swarm.specs.B[mode,:,:]
        u_unc = np.dot(np.linalg.inv(B),xf - xi)
        u_unc = np.reshape(u_unc,(2,-1)).T
        u = np.zeros_like(u_unc)
        u[:,-1] = u_unc[:,0]/n_outer
        u[:,:-1] = u_unc[:,1:]/n_outer
        
        U = np.empty((2,0),float)
        for i in range(n_mode-1):
            U = np.hstack( (U, np.tile(u[:,i][np.newaxis].T,(1,1)) ) )
        U = np.hstack( (U, u[:,-1][np.newaxis].T ) )
        U_sol = U
        for i in range(n_outer - 1):
            U_sol = np.hstack( (U_sol,U) )
        #
        X_sol = np.zeros((2*n_robot, U_sol.shape[1]))
        X_sol[:,0] = xi
        counter = 0
        for i_outer in range(n_outer):
            for mode in range(1,n_mode):
                mapped_mode = mode_sequence[mode]
                X_sol[:,counter + 1] = (X_sol[:,counter]
                        + np.dot(self.swarm.specs.B[mapped_mode,:,:],
                        U_sol[:,counter]))
                counter += 1
            if i_outer < n_outer - 1:
                mode = 0
                X_sol[:,counter + 1] = (X_sol[:,counter]
                     + np.dot(self.swarm.specs.B[mode,:,:],
                       U_sol[:,counter]))
            counter += 1
        sol = {}
        sol['x']  = np.hstack((U_sol.T.flatten(), X_sol.T.flatten()))
        return sol

    def solve_optimization(self, xf, boundary = False):
        """Solves the optimization problem, sorts and post processes the
        answer and returns the answer.
        """
        lbx, ubx = self.lbx, self.ubx
        lbg, ubg = self.lbg, self.ubg
        solver = self.solver
        xi = self.xi
        self.xf = xf

        U0 = ca.DM.zeros(self.U.shape)
        X0 = ca.DM(np.matlib.repmat(xi,self.X.shape[1],1).T)
        X0 = ca.DM.zeros(self.X.shape)
        x0 = ca.vertcat(ca.reshape(U0,-1,1), ca.reshape(X0,-1,1))
        p = np.hstack((xi, xf))

        sol = solver(x0 = x0, lbx = lbx, ubx = ubx,
                     lbg = lbg, ubg = ubg, p = p)
        #sol = solver(x0 = x0, lbg = lbg, ubg = ubg, p = p)
        # recovering the solution in appropriate format
        P_sol = np.vstack((xi,xf)).T
        if boundary is True:
            U_sol, X_sol, UZ, U, X = self.__post_process_wrap(sol)
        else:
            U_sol, X_sol, UZ, U, X = self.__post_process_u(sol)
            
        return sol, U_sol, X_sol, P_sol, UZ, U, X
        
########## test section ################################################
if __name__ == '__main__':
    a, ap = [0,0], [0,20]
    b, bp = [20,0], [20,20]
    c, cp = [40,0], [40,20]
    d, dp = [60,0], [60,20]
    e = [75,0]
    xi = np.array(a+b+c+d)
    transfer = np.array([0,0]*(len(xi)//2))
    xi = xi + transfer
    A = np.array([-15,0]+[-15,30]+[0,45]+[15,30]+ [15,0])
    F = np.array([0,0]+[0,30]+[0,50]+[25,50]+ [20,30])
    M = np.array([-30,0]+[-15,60]+[0,40]+[15,60]+ [30,0])
    xf = np.array([0,0]+[0,30]+[0,50]+[25,50])
    #xf = np.array([0,30]+[0,50]+[25,50])
    #xf = np.array(dp+cp+bp+ap)
    outer = 3
    boundary = True
    last_section = True
    
    #pivot_separation = np.array([[10,9,8,7,6],[9,8,7,6,10],[8,7,6,10,9],[7,6,10,9,8]])
    pivot_separation = np.array([[10,9,8,7],[9,8,7,10],[8,7,10,9]])
    #pivot_separation = np.array([[10,9,8],[9,8,10]])
    
    swarm_specs=model.SwarmSpecs(pivot_separation, 5, 10)
    swarm = model.Swarm(xi, 0, 1, swarm_specs)
    planner = Planner(swarm, n_outer = outer)

    #g, lbg, ubg = planner.get_constraints(boundary = boundary)
    #G = ca.Function('g',[planner.X,planner.U,planner.P],[g])
    #optim_var, lbx, ubx, p = planner.get_optim_vars(boundary=boundary)
    #obj = planner.get_objective()
    #nlp_prob = {'f': obj, 'x': optim_var, 'g': g, 'p': p}
    solver = planner.get_optimization(solver_name='knitro', boundary=False)
    sol,U_sol,X_sol,P_sol,UZ,U,X = planner.solve_optimization(xf,boundary)
    swarm.reset_state(xi,0,1)
    anim =swarm.simanimation(U,1000,boundary=boundary,last_section=last_section)
    
    swarm.reset_state(xi,0,1)
    swarm.simplot(U,500,boundary=boundary,last_section=last_section)
    #x = ca.SX.sym('x',4*2)
    #u = ca.SX.sym('u',2)
    #i = 2
    #print(planner.swarm.specs.B[i,:,:])
    #print(planner.f(x,u,i))
    #print(planner.swarm.specs.n_robot)
    #print(planner.swarm.specs.n_mode)
    #print(planner.n_inner)
    #print(planner.n_outer)