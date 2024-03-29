#%%
########################################################################
# This files hold classes and functions that plans swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from itertools import combinations
from collections import deque, Counter

import numpy as np
import numpy.matlib
import casadi as ca

try:
    from swarm import model
except ModuleNotFoundError:
    import model

np.set_printoptions(precision=2, suppress=True)
########## classes and functions #######################################
class Planner():
    """
    This class contains objects and methods for planning a path
    for a given swarm of milirobots.
    """

    def __init__(self, specs, mode_sequence = None, steps = 3,
                             solver_name='ipopt', feastol= 3.5, boundary=True):
        """
        ----------
        Parameters
        ----------
        specs: model.SwarmSpecs
            Instance of SwarmSpecs class.
        mode_sequence: list, default: None
            Arbitrary sequence of modes covering all modes uniquely.
            1D integer array.
        steps: Int, default: 3
            Number of steps to loop over modes in planning.
        """
        self.specs = specs
        self.cost_coeff = 10
        self.robot_pairs = self.specs.robot_pairs
        self.d_min = self.specs.d_min
        self.ubsx = self.specs.ubsx
        self.lbsx = self.specs.lbsx
        self.ubsy = self.specs.ubsy
        self.lbsy = self.specs.lbsy
        self.rscoil = self.specs.rscoil
        self.lbg, self.ubg = [None]*2
        self.lbx, self.ubx = [None]*2
        self.solver_opt = self._solvers(feastol)
        self.boundary = boundary
        self.solver_name = solver_name
        # Build mode sequence
        self._set_mode_sequence(mode_sequence, steps)
        # Build the optimization problem.
        self._get_optimization(solver_name, boundary)
    
    def _set_mode_sequence(self, mode_sequence, steps):
        """
        Checks mode_sequence, if available.
        Otherwise returns the default sequence.
        """
        # Modify tumble_index.
        n_mode = self.specs.n_mode
        if mode_sequence is None:
            mode_sequence = list(range(1,n_mode)) + [0]
        self.mode_sequence = deque(mode_sequence*steps)
        # Calculate next mode sequence.
        next_modes= deque([mode for mode in self.mode_sequence if mode])
        next_modes.rotate(-1)
        next_mode_sequence = []
        cnt= 0
        for mode in self.mode_sequence:
            if mode:
                next_mode_sequence.append(next_modes[cnt])
                cnt+= 1
            else:
                next_mode_sequence.append(0)
        self.next_mode_sequence = deque(next_mode_sequence)
    
    def _f(self,state,control,mode):
        """Defines swarm transition for CASADI."""
        if mode < 0:
            mode = 0 # Mode change uses tumbling control matrix.
        B = self.specs.B
        next_state = state + ca.mtimes(B[int(mode),:,:],control)
        return next_state.full().squeeze()
    
    def _fcasadi(self,state,control,mode):
        """Defines swarm transition for CASADI."""
        B = self.specs.B
        next_state = state + ca.mtimes(B[int(mode),:,:],control)
        return next_state
    
    def _construct_vars(self):
        """
        Constructing optimization variables.
        """
        mode_seq= self.mode_sequence
        n_robot = self.specs.n_robot
        n_mode_seq = len(mode_seq)
        U = ca.SX.sym('u',2,n_mode_seq) # U_step_cmdmode
        U0 = ca.SX.sym('u0',2,n_mode_seq) # U_step_cmdmode
        BC = ca.SX.sym('bc',2*n_robot,2) # Start and end point.
        counter = 0
        # Build X and U.
        # Construct position and control for current outer loop.
        for idx, mode in enumerate(mode_seq):
            # Maps to current mode sequence.
            varstr = f'{idx:02d}_{mode:01d}'
            # Construct current input.
            U[0,idx] = ca.SX.sym('ux_'+varstr)
            U[1,idx] = ca.SX.sym('uy_'+varstr)
            U0[0,idx] = ca.SX.sym('u0x_'+varstr)
            U0[1,] = ca.SX.sym('u0y_'+varstr)
        # Building P
        for robot in range(n_robot):
            BC[2*robot,0] = ca.SX.sym('xi_{:02d}'.format(robot))
            BC[2*robot + 1,0] = ca.SX.sym('yi_{:02d}'.format(robot))
            BC[2*robot,1] = ca.SX.sym('xf_{:02d}'.format(robot))
            BC[2*robot + 1,1] = ca.SX.sym('yf_{:02d}'.format(robot))
        # Obnjective activation
        obj_act= ca.SX.sym('obj_act')
        return U, U0, BC, obj_act

    def _get_constraint_inter_robot(self,x,u,mode):
        """
        Returns inter-robot constraints.
        It uses u = [r, theta] as input.
        Look at the paper for derivation of the constraint.
        """
        g = []
        dm = self.d_min
        beta = self.specs.beta[mode,:]
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
                   *(-b*dx+a*dy)/dm)]#,
                  #a**2+b**2-dm**2]
        return g
    
    def _get_constraint_shooting(self,x_next, x, u, mode):
        """Returns constraint resulted from multiple shooting."""
        B = self.specs.B[mode,:,:]
        g = []
        g += [x_next - x - ca.mtimes(B,u)]
        return g
    
    def _get_constraint_distance(self,x_next):
        """
        This function return the constraint with respective to
        maximum distance between each milirobot for each axis.
        """
        g = []
        for pair in self.robot_pairs:
            zi = x_next[2*pair[0]:2*pair[0]+2]
            zj = x_next[2*pair[1]:2*pair[1]+2]
            g += [zi-zj]
        return g
    
    def _get_constraint_coil(self, x):
        """Returns constraint due to smallest coil."""
        radii = self.rscoil
        n_robot = self.specs.n_robot
        g = []
        for robot in range(n_robot):
            g += [radii**2 - x[2*robot]**2 - x[2*robot+1]**2]
        return g

    def _get_constraints(self, boundary = False):
        """This function builds constraints of optimization."""
        n_robot = self.specs.n_robot
        mode_seq = self.mode_sequence
        n_mode_seq = len(mode_seq)
        bc_tol = self.specs.bc_tol
        lbsx, ubsx= self.lbsx, self.ubsx
        lbsy, ubsy= self.lbsy, self.ubsy
        U = self.U
        BC = self.BC
        g_shooting = []
        g_terminal = []
        g_inter_robot = []
        g_distance = []
        g_coil = []
        state = BC[:,0]
        for idx, mode in enumerate(mode_seq):
            control = U[:,idx]
            state_next = self._fcasadi(state, control, mode)
            if idx:
                g_shooting += [state]
                g_coil += self._get_constraint_coil(state)
            if mode != 0:
                # Tumbling, no need to the constraints.
                g_inter_robot += self._get_constraint_inter_robot(state,
                                                                    control,
                                                                    mode)
                g_distance += self._get_constraint_distance(state_next)
            state = state_next
        # Add shooting constraint of terminal position.
        g_terminal += [BC[:,1] - state]
        # Configure bounds of g
        n_g_shooting = ca.vertcat(*g_shooting).shape[0]
        n_g_terminal = ca.vertcat(*g_terminal).shape[0]
        n_g_inter_robot = ca.vertcat(*g_inter_robot).shape[0]
        n_g_distance = ca.vertcat(*g_distance).shape[0]
        n_g_coil = ca.vertcat(*g_coil).shape[0]
        #
        g = ca.vertcat(*(g_terminal 
                       + g_inter_robot 
                       ))
        lbg = np.hstack((-bc_tol*np.ones(n_g_terminal),
                         np.zeros(n_g_inter_robot),
                          ))
        ubg = np.hstack((bc_tol*np.ones(n_g_terminal),
                         np.inf*np.ones(n_g_inter_robot),
                          ))
        if boundary:
            g = ca.vertcat(g,*g_coil)
            lbg = np.hstack((lbg,np.zeros(n_g_coil)))
            ubg = np.hstack((ubg,np.inf*np.ones(n_g_coil)))
            g= ca.vertcat(g,*g_shooting)
            lbg = np.hstack((lbg,[lbsx,lbsy]*n_robot*(n_mode_seq-1)))
            ubg = np.hstack((ubg,[ubsx,ubsy]*n_robot*(n_mode_seq-1)))
        return g, lbg, ubg

    def _get_optim_vars(self,boundary = False):
        """This function returns optimization flatten variable and its
        bounds."""
        U = self.U
        U0 = self.U0
        BC  = self.BC
        obj_act = self.obj_act
        n_mode_seq = len(self.mode_sequence)
        width = self.ubsx - self.lbsx
        height = self.ubsy - self.lbsy
        optim_var = ca.reshape(U,-1,1)
        # Configure bounds.
        if boundary is True:
            # Bounds of U
            lbu = [-width, -height]
            ubu = [ width,  height]
        else:
            # Bounds of U
            lbu = [-np.inf, -np.inf]
            ubu = [ np.inf,  np.inf]
        # concatenating X and U bounds
        lbx = np.array((lbu)*n_mode_seq)
        ubx = np.array((ubu)*n_mode_seq)
        # concatenating optimization parameter
        p = ca.vertcat(ca.reshape(BC,-1,1),
                       ca.reshape(U0,-1,1), obj_act)
        return optim_var, lbx, ubx, p

    def _get_objective(self):
        """Returns objective function for optimization.
        If sparse = True, then it returns first norm objective function
        that favors sparsity.
        """
        n_mode_seq = len(self.mode_sequence)
        n_mode= self.specs.n_mode
        U = self.U #- self.U0
        obj_act= self.obj_act
        obj = 0
        for idx, mode in enumerate(range(n_mode_seq)):
            u = U[:,idx]
            obj+= (idx//n_mode)**0*ca.sum1(u*u)#(idx//n_mode)**2*ca.norm_inf(u)#
        """ for i in range(U.shape[1]):
            u = U[:,i]
            obj += ca.sum1(u*u) """
        return obj*obj_act

    def _get_optimization(self, solver_name = 'ipopt', boundary = False):
        """Sets up and returns a CASADI optimization object."""
        # Construct optimization symbolic vars in CASADI.
        self.U, self.U0, self.BC, self.obj_act = self._construct_vars()
        # Construct all constraints.
        g, lbg, ubg = self._get_constraints(boundary)
        self.g, self.lbg, self.ubg = g, lbg, ubg
        # Concatenate optimization variables and set bounds.
        optim_var, lbx, ubx, p = self._get_optim_vars(boundary)
        self.optim_var = optim_var
        self.lbx, self.ubx = lbx, ubx
        # Build objective function.
        obj = self._get_objective()
        # Construct the solver.
        nlp_prob = {'f': obj, 'x': optim_var, 'g': g, 'p': p}
        _solver = self.solver_opt[solver_name]
        solver = ca.nlpsol('solver',_solver['name'],nlp_prob,_solver['opts'])
        self.solver = solver
        return solver

    def _solvers(self, feastol):
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
        
        # Debuging info level.
        solvers['knitro']['opts']['knitro.debug'] = 0
        solvers['knitro']['opts']['knitro.alg'] = 3
        #solvers['knitro']['opts']['knitro.ma_terminate'] = 1
        # Printing level.
        #solvers['knitro']['opts']['knitro.outlev'] = 0
        #solvers['knitro']['opts']['knitro.outmode'] = 0
        # Max time for each start point in seconds
        solvers['knitro']['opts']['knitro.maxtime_real'] = 3
        # Choose multistart.
        solvers['knitro']['opts']['knitro.multistart'] = 1
        # Max multistart tries.
        solvers['knitro']['opts']['knitro.ms_maxsolves'] = 200
        # Stop at first feasible solution.
        solvers['knitro']['opts']['knitro.ms_terminate'] = 2
        # Enforce feasibility search over optimality.
        solvers['knitro']['opts']['knitro.bar_feasible'] = 2
        # Switch to feasibility more aggresively.
        solvers['knitro']['opts']['knitro.bar_switchrule'] = 3
        # How many fine iteration to do after interior point method.
        solvers['knitro']['opts']['knitro.bar_maxcrossit'] = 30
        # Feasibility tolerance
        solvers['knitro']['opts']['knitro.feastol'] = 1000
        # This will be enforced.
        solvers['knitro']['opts']['knitro.feastolabs'] = feastol
        return solvers

    @staticmethod
    def _cartesian_to_polar(z):
        """Converts cartesian to polar coordinate."""
        z = complex(z[0],z[1])
        z = np.array([np.abs(z), np.angle(z)])
        return z

    def _accurate_rotation(self,u):
        """This function gets a desired rotation and returns a sequence
        of two pure steps of rotations that produce the desired movement
        in rotation mode."""
        tumbling_length = self.specs.tumbling_length
        r = np.linalg.norm(u)
        if r>1.0:
            # Practially we cannot execute such small displacements.
            # Divide r into even number of tumblings.
            r1 = np.ceil(.5*r/tumbling_length)*(2*tumbling_length)
            r1 -= tumbling_length
            r2 = tumbling_length
            # Calculate angles so the two complete tumbling motions
            # be equivalent of original tumbling requested.
            teta = ca.SX.sym('t',2)
            f1 = r1*ca.cos(teta[0])+r2*ca.cos(teta[1]) - u[0]
            f2 = r1*ca.sin(teta[0])+r2*ca.sin(teta[1]) - u[1]
            f = ca.Function('g',[teta],[ca.vertcat(*[f1,f2])])
            F = ca.rootfinder('F','newton',f)
            teta_value = F(np.random.rand(2))
            u_possible = np.zeros((2,2))
            u_possible[0,0] = r1*np.cos(teta_value[0])
            u_possible[1,0] = r1*np.sin(teta_value[0])
            u_possible[0,1] = r2*np.cos(teta_value[1])
            u_possible[1,1] = r2*np.sin(teta_value[1])
        else:
            u_possible = np.zeros((2,2))
        return u_possible

    def _post_process_u(self,sol):
        """Post processes the solution and adds intermediate steps."""
        mode_seq = self.mode_sequence
        n_mode_seq_nz= np.count_nonzero(mode_seq)
        n_mode_seq= len(mode_seq)
        mode_start = mode_seq[np.nonzero(mode_seq)[0][0]]
        n_robot = self.specs.n_robot
        xi = self.xi
        UZ_raw = ca.reshape(sol['x'],2,-1)[:2,:].full()
        UZ_raw= np.vstack((UZ_raw,mode_seq))
        U_raw= np.zeros_like(UZ_raw)
        r_raw= np.zeros_like(UZ_raw)
        r_raw[2,:]= mode_seq
        X_raw = np.zeros((2*n_robot, n_mode_seq + 1))
        X_raw[:,0]= xi
        UZ = np.zeros((3,2*n_mode_seq))
        X = np.zeros((2*n_robot, 2*n_mode_seq + 1))
        X[:,0] = xi
        U = np.zeros_like(UZ)
        # Adjust mode_change parameters.
        dir_mode = 1 if mode_start%2 else -1
        mode_change = np.array([self.specs.mode_rel_length[1], 0.0])*dir_mode
        mode_change_remainder  = ((n_mode_seq_nz)%2)*mode_change
        # Recovering input with transitions
        counter = 0
        for idx, mode in enumerate(mode_seq):
            if mode == 0:
                # This tumbling.
                U_tumbled = self._accurate_rotation(
                    UZ_raw[:2,idx] - mode_change_remainder)
                UZ[:2,counter:counter+2] = U_tumbled
                UZ[2,counter:counter+2] = mode
            else:
                # This is pivot_walking and needs mode change.
                UZ[:2,counter] = UZ_raw[:2, idx]
                UZ[2,counter] = mode
                # Calculate next mode and perform it.
                next_mode = self.next_mode_sequence[idx]
                UZ[:2,counter+1] = mode_change
                UZ[2,counter+1] = -next_mode
                mode_change *= -1
            r_raw[:2,idx]= self._cartesian_to_polar(UZ_raw[:2,idx])
            # Calculate positions.
            X_raw[:,idx+1]= self._f(X_raw[:, idx], UZ_raw[:2, idx], mode)
            X[:,counter+1] = self._f(X[:,counter], UZ[:2,counter],
                                                                UZ[2,counter])
            X[:,counter+2] = self._f(X[:,counter+1], UZ[:2,counter+1],
                                                            UZ[2,counter+1])
            counter += 2
        # Calculate the corresponding polar coordinate inputs.
        U[2,:] = UZ[2,:]
        for i in range(UZ.shape[1]):
            U[:2,i] = self._cartesian_to_polar(UZ[:2,i])
            if U[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                U[0,i] = int(U[0,i]*1000)/1000
        U_raw[2,:] = UZ_raw[2,:]
        for i in range(UZ_raw.shape[1]):
            U_raw[:2,i] = self._cartesian_to_polar(UZ_raw[:2,i])
            if U_raw[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                U_raw[0,i] = int(U_raw[0,i]*1000)/1000
        return UZ_raw, U_raw, r_raw, X_raw, UZ,  U, X

    def solve_unconstrained(self,xi,xf):
        """This function solved the unconstrained case from
        controllability analysis and adjusts it for the problem setting,
        to be used as initial guess."""
        n_mode = self.specs.n_mode
        n_robot = self.specs.n_robot
        B = np.zeros((2*n_robot,2*n_robot))
        for mode in range(n_robot):
            B[:,2*mode:2*mode +2] = self.specs.B[mode,:,:]
        UUZ_raw = np.dot(np.linalg.inv(B),xf - xi)
        if n_robot< n_mode:
            UUZ_raw= np.append(UUZ_raw, (n_mode-n_robot)*[0.0,0.0])
        UUZ_raw = np.reshape(UUZ_raw,(-1,2)).T  
        UUZ_raw= np.roll(np.vstack((UUZ_raw, range(n_mode))),-1,axis= 1)
        XU_raw= np.zeros((2*n_robot,n_mode+1))
        XU_raw[:,0]= xi
        XU= np.zeros((2*n_robot,2*n_mode+1))
        XU[:,0]= xi
        mode_seq= UUZ_raw[2,:].astype(int)
        next_mode_seq= np.append(np.roll(mode_seq[mode_seq !=0], -1),0)
        UUZ= np.zeros((3,2*n_mode),dtype= float)
        # Adjust mode_change parameters.
        dir_mode = 1 
        mode_change = np.array([self.specs.mode_rel_length[1], 0.0])*dir_mode
        mode_change_remainder  = ((n_mode-1)%2)*mode_change
        counter= 0
        for idx, mode in enumerate(mode_seq):
            if mode == 0:
                # This tumbling.
                U_tumbled = self._accurate_rotation(
                                UUZ_raw[:2, idx] - mode_change_remainder)
                UUZ[:2,counter:counter+2] = U_tumbled
                UUZ[2,counter:counter+2] = mode
            else:
                # This is pivot_walking and needs mode change.
                UUZ[:2,counter] = UUZ_raw[:2,idx]
                UUZ[2,counter] = mode
                # Calculate next mode and perform it.
                next_mode = next_mode_seq[idx]
                UUZ[:2,counter+1] = mode_change
                UUZ[2,counter+1] = -next_mode
                mode_change *= -1
            # Calculate positions.
            XU_raw[:,idx+1]= self._f(XU_raw[:,idx], UUZ_raw[:2,idx],
                                                    UUZ_raw[2,idx])
            XU[:,counter+1] = self._f(XU[:,counter], UUZ[:2,counter],
                                                     UUZ[2,counter])
            XU[:,counter+2] = self._f(XU[:,counter+1], UUZ[:2,counter+1],
                                                       UUZ[2,counter+1])
            counter += 2
        # Add polar versions
        UU= np.zeros_like(UUZ)
        UU[2,:] = UUZ[2,:]
        for i in range(UUZ.shape[1]):
            UU[:2,i] = self._cartesian_to_polar(UUZ[:2,i])
            if UU[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                UU[0,i] = int(UU[0,i]*1000)/1000
        UU_raw= np.zeros_like(UUZ_raw)
        UU_raw[2,:] = UUZ_raw[2,:]
        for i in range(UUZ_raw.shape[1]):
            UU_raw[:2,i] = self._cartesian_to_polar(UUZ_raw[:2,i])
            if UU_raw[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                UU_raw[0,i] = int(UU_raw[0,i]*1000)/1000
        return UUZ_raw, UU_raw, XU_raw, UUZ,  UU, XU
    
    def _isfeasible(self,UX,g):
        """
        Determines if the solution is feasible.
        Casadi does not provide access to solution status.
        """
        flag = True
        flag_value = 0
        flag_tol = self.solver_opt['knitro']['opts']['knitro.feastolabs']
        # Check upper and lower bounds of solution.
        flag_value = min(flag_value, min(UX - self.lbx))
        flag_value = min(flag_value, min(self.ubx - UX))
        # Check upper and lower bounds of constraints.
        flag_value = min(flag_value, min(g - self.lbg))
        flag_value = min(flag_value, min(self.ubg - g))
        #
        if abs(flag_value)> flag_tol:
            flag = False
        return flag
    
    def _isfeasible_from_unconstrained(self, UUZ_raw):
        UUZ_raw= np.roll(UUZ_raw,1, axis=1)
        width = self.ubsx - self.lbsx
        height = self.ubsy - self.lbsy
        counts= Counter(self.mode_sequence)
        flag= True
        for i, count in counts.items():
            if (   (abs(UUZ_raw[0,i]) > width/count)
                or (abs(UUZ_raw[1,i]) > height/count)):
                flag= False
                print(f'Might be infeasible in mode {i}, at {count} steps.')
                print("Increase the steps.")
                break
        return flag
    
    def _threshold_mode_sequence(self, UZ_raw, r_raw):
        """
        This function looks into a given raw solution, threshold it
        and returns a new mode_sequence withouth thresholded modes.
        The purpose of this function is to reduce unnecessary steps.
        """
        threshold= max(self.specs.tumbling_length,15.0)
        UZ_raw= UZ_raw[:,r_raw[0,:]> threshold]
        mode_sequence= UZ_raw[2,:].astype(int)
        return UZ_raw, mode_sequence

    def _solve(self, xi, xf, U0, obj_act= 1):
        """
        Low level function for solving the optimization.
        """
        lbx, ubx = self.lbx, self.ubx
        lbg, ubg = self.lbg, self.ubg
        self.xi, self.xf= xi, xf
        solver = self.solver
        UX0 = ca.reshape(U0[:2,:],-1,1)
        p = ca.vertcat(xi, xf, ca.reshape(U0[:2,:],-1,1),obj_act)
        sol = solver(x0 = UX0, lbx = lbx, ubx = ubx,
                     lbg = lbg, ubg = ubg, p = p)
        UX_raw = sol['x'].full().squeeze()
        g_raw = sol['g'].full().squeeze()
        isfeasible = self._isfeasible(UX_raw,g_raw)
        return sol, isfeasible

    def solve(self, xi, xf, resolve= True, threshold= False):
        """
        Solves the optimization problem, sorts and post processes the
        answer and returns the answer.
        ----------
        Parameters
        ----------
        xi: Numpy nd.array
            Initial position of robots.
        xf: Numpy nd.array
            Final position of robots.
        """
        obj_act= 0 if resolve else 1.0
        # Solve unconstrained
        UUZ_raw,UU_raw, XU_raw, UUZ,  UU, XU= self.solve_unconstrained(xi, xf)
        self._isfeasible_from_unconstrained(UUZ_raw)
        # Solve for feasibility with constant objective function.
        U0 = ca.DM.zeros(self.U.shape)
        UX0 = ca.reshape(U0[:2,:],-1,1)
        sol, isfeasible= self._solve(xi,xf, U0, obj_act= obj_act)
        BC = np.vstack((xi,xf)).T
        UZ_raw, U_raw, r_raw, X_raw, UZ,  U, X= self._post_process_u(sol)
        print(r_raw.T)
        if resolve and isfeasible:
            if threshold:
                U0, mode_sequence= self._threshold_mode_sequence(UZ_raw, r_raw)
                # Build mode sequence
                self._set_mode_sequence(mode_sequence, 1)
                # Build the optimization problem.
                self._get_optimization(self.solver_name, self.boundary)
            else:
                U0= UZ_raw
            # Solve with non_constant objective.
            sol_n, isfeasible_n= self._solve(xi,xf, U0, obj_act= 1.0)
            if isfeasible_n:
                isfeasible= isfeasible_n
                sol= sol_n
                UZ_raw,U_raw,r_raw,X_raw,UZ,U,X= self._post_process_u(sol)
                print(r_raw.T)
        return sol,UZ_raw, U_raw, X_raw, BC, UZ, U, X, isfeasible
    
    @classmethod
    def plan(cls,xg, outer_steps, mode_sequence, specs, feastol,
                                resolve= True,boundary=True, threshold= False):
        """
        Wrapper for planning class simplifying creation and calling
        process.
        ----------
        Parameters
        ----------
        xg: array of final position [x_i, y_i, ...]
        outer_steps: minimum value of outer steps to be used.
        mode_sequence: list, sequence to be used.
        specs: instance of model.SwarmSpecs containing swarm info.
        ----------
        Yields
        ----------
        polar_cmd: 2D array of polar input  [[r, phi, mode], ... ]
        """
        # Build planner
        planner = cls(specs, mode_sequence= mode_sequence, steps= outer_steps,
                     solver_name='knitro', feastol= feastol, boundary=boundary)
        # Get new initial condition.
        xi = yield None
        _, UZ_raw, _, _, _, _, U, _, isfeasible= planner.solve(xi,xg,resolve,
                                                                 threshold)
        print(f"{isfeasible = } at {outer_steps:01d} outer_steps.")
        if not isfeasible:
            raise RuntimeError
        yield U.T
        return planner

def main3p():
    outer = 4
    boundary = True
    last_section = False

    #pivot_length = np.array([[10,9,8],[9,8,10]])
    mode_sequence= [0,1,2]
    #specs=model.SwarmSpecs(pivot_length,10)
    specs = model.SwarmSpecs.robo(3)
    specs.set_space(ubx=-3, lbx= -115, rcoil= 100)
    xi = specs.get_letter('e', ang= 0, roll= 0)[0]
    xf = specs.get_letter('f', ang= 0, roll= 0)[0]
    s1 = np.array([-30,  0,   0,  0, +30,  0],dtype=float)
    s2 = np.array([+30,+30,   0,  0,   0,+30],dtype=float)
    s3 = np.array([-30,  0, -30,-30,   0,  0],dtype=float)
    s4 = np.array([  0,  0,   0,-30, -30,-30],dtype=float)
    s5 = np.array([+30,-30,   0,  0,  30,  0],dtype=float)
    s6 = np.array([  0,  0, +30,  0, -30,  0],dtype=float)
    #xi = s2*4/3
    #xf = s5*4/3
    xi = np.array([+40,+30, +40,  0, +40,-30],dtype=float)
    xf = np.array([+25,+10, +55,-10, +85,+10],dtype=float)
    xi = np.array([-70,+10, -40,-10, -10,+10],dtype=float)
    xf = np.array([-40,+30, -40,  0, -40,-30],dtype=float)
    #xf = np.array([+026.20,+011.02,+054.60,-010.77,+083.74,+009.79],dtype=float)
    #xi = np.array([-065.13,+012.24,-039.18,-006.37,-008.81,+013.71],dtype=float)
    
    swarm = model.Swarm(xi, 0, 1, specs)
    #planner = Planner(specs, mode_sequence , steps = outer,
    #                        solver_name='knitro', boundary=boundary)
    #g, lbg, ubg = planner._get_constraints()
    #optim_var, lbx, ubx, p = planner._get_optim_vars(boundary)
    #obj = planner._get_objective()
    #solver = planner._get_optimization(solver_name='knitro', boundary=boundary)
    #sol, UZ_raw, U_raw, X_raw, P, UZ, U, X, isfeasible = planner.solve(xi, xf)
    #print(isfeasible)
    resolve = True
    planner = Planner.plan(xf,outer,mode_sequence,specs,3.5, resolve,boundary)
    U = next(planner)
    if U is None:
        U = planner.send(xi)
        print(U)
        U = U.T
    swarm.reset_state(xi,0,1)
    #anim =swarm.simanimation(U,1000,boundary=boundary,last_section=last_section,save = False)
    swarm.reset_state(xi,0,1)
    swarm.simplot(U,1000,boundary=boundary,last_section=last_section)

def main4p():
    outer = 1
    boundary = True
    last_section = False
    mode_sequence= [0,1,2,3,4]*4
    specs= model.SwarmSpecs.robo(4)
    specs.set_space(ubx= -3, lbx= -115, rcoil= 100)
    scale= 1#3.0/4.0
    xi = specs.get_letter('*', ang= 0, roll= 0)[0]*scale
    xf = specs.get_letter('Z', ang= 0, roll= 0)[0]*scale
    A = np.array([-15,0]+[-15,30]+[0,60]+[15,30]+ [15,0])
    A= np.array([-40,-40, -40,  0,   0, 40,  40,  0,  40,-40],dtype= float)
    F = np.array([0,0]+[0,30]+[0,50]+[25,50]+ [20,30])
    M = np.array([-30,0]+[-15,60]+[0,40]+[15,60]+ [30,0])
    Y= np.array([ 0,0.0, 0,-50, 30,40, -30,40])
    xi = np.array([+40,+45, +40,+15, +40,-15, +40,-45],dtype=float)
    xf = np.array([+10,+10, +30,-10, +50,+10, +70,-10],dtype=float)
    xi = np.array([-72,+10, -52,-10, -32,+10, -12,-10],dtype=float)
    xf = np.array([-40,+45, -40,+15, -40,-15, -40,-45],dtype=float)
    #xf= Y
    #xf= np.array([20.0,0.0, -20.0,0, -40,0, 40,0])
    print(xi)
    print(xf)
    swarm = model.Swarm(xi, 0, 1, specs)
    resolve = True
    threshold= False
    planner = Planner.plan(xf,outer,mode_sequence,specs,3.5, resolve,boundary,threshold)
    U = next(planner)
    if U is None:
        U = planner.send(xi)
        print(U)
        U = U.T
    swarm.reset_state(xi,0,1)
    #anim =swarm.simanimation(U,1000,boundary=boundary,last_section=last_section,save = False)
    swarm.reset_state(xi,0,1)
    swarm.simplot(U,1000,boundary=boundary,last_section=last_section)

def main5p():
    outer = 1
    boundary = True
    last_section = False
    mode_sequence= [0,1,2,3,4]*4
    specs = model.SwarmSpecs.robo(5)
    scale= 1.0#3.0/4.0
    xi = specs.get_letter('*', ang= 0, roll= 0)[0]*scale
    xf = specs.get_letter('R', ang= 0, roll= 0)[0]*scale
    #xi= np.array([-033.91,+030.73,-002.33,+045.42,+033.42,+028.77,-030.12,-042.60,+032.24,-037.84])
    #xi= np.array([-031.37,-034.74,-030.59,+000.13,-027.67,+038.20,+019.83,+018.12,+028.40,-041.87])
    print(xi)
    print(xf)
    swarm = model.Swarm(xi, 0, 1, specs)
    resolve = True
    threshold= False
    planner = Planner.plan(xf,outer,mode_sequence,specs,3.5, resolve,boundary, threshold)
    U = next(planner)
    if U is None:
        U = planner.send(xi)
        print(U)
        U = U.T
    swarm.reset_state(xi,0,1)
    #anim =swarm.simanimation(U,1000,boundary=boundary,last_section=last_section,save = False)
    swarm.reset_state(xi,0,1)
    swarm.simplot(U,1000,boundary=boundary,last_section=last_section)

def main5o():
    outer = 1
    boundary = True
    last_section = False
    mode_sequence= [0,1,2,3,4]*3
    a= 8.0
    b= 5.0
    pivot_length = np.array([[b, a, a, b, b],
                             [a, a, b, b, b],
                             [a, b, b, b, a],
                             [b, b, b, a, a]])
    pivot_length = np.array([[4.80,7.90,7.90,4.90,4.80],
                             [8.00,8.00,4.90,4.90,4.90],
                             [8.00,4.90,4.90,5.00,8.10],
                             [5.00,5.00,5.10,8.00,8.20]])

    specs = model.SwarmSpecs(pivot_length,15.24)
    #specs= model.SwarmSpecs.robo5p()
    #specs.ubx = 115*1.0
    #specs.uby = 90*1.0
    #specs.lbx = -specs.ubx
    #specs.lby = -specs.uby
    specs.d_min= 20
    #specs.rcoil = 95*1.0
    #specs.ubsx = specs.ubx - (specs.tumbling_length+4)#*1.5
    #specs.lbsx = -specs.ubsx
    #specs.ubsy = specs.uby - 10#specs.tumbling_length*1.1
    #specs.lbsy = -specs.ubsy
    #specs.rscoil = specs.rcoil - specs.tumbling_length
    scale= 1.0
    xi = specs.get_letter('*', ang= 0, roll= 0)[0]#*scale
    xf= specs.get_letter('S', ang= 0, roll= 0)[0]
    A = np.array([-15,0]+[-15,30]+[0,60]+[15,30]+ [15,0])
    A= np.array([-40,-40, -40,  0,   0, 40,  40,  0,  40,-40],dtype= float)
    F = np.array([0,0]+[0,30]+[0,50]+[25,50]+ [20,30])
    M = np.array([-30,0]+[-15,60]+[0,40]+[15,60]+ [30,0])
    #xf= A
    print(xi)
    print(xf)
    swarm = model.Swarm(xi, 0, 1, specs)
    resolve = True
    planner = Planner.plan(xf,outer,mode_sequence,specs,3.5, resolve,boundary)
    U = next(planner)
    if U is None:
        U = planner.send(xi)
        print(U)
        U = U.T
    swarm.reset_state(xi,0,1)
    #anim =swarm.simanimation(U,1000,boundary=boundary,last_section=last_section,save = False)
    swarm.reset_state(xi,0,1)
    swarm.simplot(U,1000,boundary=boundary,last_section=last_section)
########## test section ################################################
if __name__ == '__main__':
    try:
        main3p()
        pass
    except RuntimeError:
        pass
