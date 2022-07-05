#%%
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
    """
    This class contains objects and methods for planning a path
    for a given swarm of milirobots.
    """

    def __init__(self, specs, mode_sequence = None, steps = 3,
                                          solver_name='ipopt', boundary=True):
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
        self.steps = steps
        self.ubsx = self.specs.ubsx
        self.lbsx = self.specs.lbsx
        self.ubsy = self.specs.ubsy
        self.lbsy = self.specs.lbsy
        self.rscoil = self.specs.rscoil
        self.lbg, self.ubg = [None]*2
        self.lbx, self.ubx = [None]*2
        self.solver_opt = self._solvers()
        self.boundary = boundary
        self.solver_name = solver_name
        # Build mode sequence
        self._set_mode_sequence(mode_sequence)
        # Build the optimization problem.
        self._get_optimization(solver_name, boundary)
    
    def _set_mode_sequence(self, mode_sequence):
        """
        Checks mode_sequence, if available.
        Otherwise returns the default sequence.
        """
        # Modify tumble_index.
        n_mode = self.specs.n_mode
        if mode_sequence is None:
            mode_sequence = list(range(1,n_mode)) + [0]
        self.mode_sequence = deque(mode_sequence)
        # Calculate next mode sequence.
        index_tumbling = self.mode_sequence.index(0)
        next_mode_sequence = self.mode_sequence.copy()
        next_mode_sequence.remove(0)
        next_mode_sequence.rotate(-1)
        next_mode_sequence.insert(index_tumbling,0)
        self.next_mode_sequence = next_mode_sequence
    
    def _f(self,state,control,mode):
        """Defines swarm transition for CASADI."""
        if mode < 0:
            mode = 0 # Mode change uses tumbling control matrix.
        B = self.specs.B
        next_state = state + ca.mtimes(B[int(mode),:,:],control)
        return next_state.full().squeeze()
    
    def _construct_vars(self):
        """
        Constructing optimization variables.
        """
        steps = self.steps
        n_mode = self.specs.n_mode
        n_robot = self.specs.n_robot
        mode_sequence = self.mode_sequence
        U = ca.SX.sym('u',2,steps*n_mode) # U_step_cmdmode
        # Xs are positions after corresponding input is applied.
        X = ca.SX.sym('x',2*n_robot,steps*n_mode) # X_step_cmdmode_robot
        BC = ca.SX.sym('bc',2*n_robot,2) # Start and end point.
        counter = 0
        # Build X and U.
        for step in range(steps):
            # Construct position and control for current outer loop.
            for mode in range(n_mode):
                # Maps to current mode sequence.
                mode = mode_sequence[mode]
                varstr = f'{step:02d}_{mode:02d}'
                # Construct current input.
                U[0,counter] = ca.SX.sym('ux_'+varstr)
                U[1,counter] = ca.SX.sym('uy_'+varstr)
                # Construct positions after applying current input.
                for robot in range(n_robot):
                    rob_str = f'_{robot:02d}'
                    X[2*robot,counter] = ca.SX.sym('x_'+varstr+rob_str)
                    X[2*robot + 1,counter] = ca.SX.sym('y_'+varstr+rob_str)
                counter += 1
        # Building P
        for robot in range(n_robot):
            BC[2*robot,0] = ca.SX.sym('xi_{:02d}'.format(robot))
            BC[2*robot + 1,0] = ca.SX.sym('yi_{:02d}'.format(robot))
            BC[2*robot,1] = ca.SX.sym('xf_{:02d}'.format(robot))
            BC[2*robot + 1,1] = ca.SX.sym('yf_{:02d}'.format(robot))
        return X, U, BC

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
                   *(-b*dx+a*dy)/dm),
                  a**2+b**2-dm**2]
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
        mode_sequence = self.mode_sequence
        n_mode = self.specs.n_mode
        bc_tol = self.specs.bc_tol
        steps = self.steps
        X = self.X
        U = self.U
        BC = self.BC
        g_shooting = []
        g_terminal = []
        g_inter_robot = []
        g_distance = []
        g_coil = []
        counter = 0
        state = BC[:,0]
        for _ in range(steps):
            for mode in range(0,n_mode):
                # Maps to mode sequence
                mode_mapped = mode_sequence[mode]
                control = U[:,counter]
                state_next = X[:,counter]
                g_shooting += self._get_constraint_shooting(state_next,
                                                            state,
                                                            control,
                                                            mode_mapped)
                if mode_mapped != 0:
                    # Tumbling, no need to the constraints.
                    g_inter_robot += self._get_constraint_inter_robot(state,
                                                                   control,
                                                                   mode_mapped)
                    g_distance += self._get_constraint_distance(state_next)
                g_coil += self._get_constraint_coil(state_next)
                state = state_next
                counter += 1
        # Add shooting constraint of terminal position.
        g_terminal += [BC[:,1] - state]
        # Configure bounds of g
        n_g_shooting = ca.vertcat(*g_shooting).shape[0]
        n_g_terminal = ca.vertcat(*g_terminal).shape[0]
        n_g_inter_robot = ca.vertcat(*g_inter_robot).shape[0]
        n_g_distance = ca.vertcat(*g_distance).shape[0]
        n_g_coil = ca.vertcat(*g_coil).shape[0]
        #
        g = ca.vertcat(*(g_shooting + g_terminal + g_inter_robot))
        lbg = np.hstack((np.zeros(n_g_shooting),
                         -bc_tol*np.ones(n_g_terminal),
                         np.zeros(n_g_inter_robot) ))
        ubg = np.hstack((np.zeros(n_g_shooting),
                         bc_tol*np.ones(n_g_terminal),
                         np.inf*np.ones(n_g_inter_robot) ))
        if boundary:
            g = ca.vertcat(g,*g_coil)
            lbg = np.hstack((lbg,np.zeros(n_g_coil)))
            ubg = np.hstack((ubg,np.inf*np.ones(n_g_coil)))
        return g, lbg, ubg

    def _get_optim_vars(self,boundary = False):
        """This function returns optimization flatten variable and its
        bounds."""
        U = self.U
        X = self.X
        BC  = self.BC
        steps = self.steps
        n_robot = self.specs.n_robot
        n_mode = self.specs.n_mode
        width = self.ubsx - self.lbsx
        height = self.ubsy - self.lbsy
        optim_var = ca.reshape(ca.vertcat(U,X),-1,1)
        # Configure bounds.
        if boundary is True:
            # Bounds of U
            lbu = [-width, -height]
            ubu = [ width,  height]
            # Bounds on X
            lbxx = [self.lbsx,self.lbsy]*n_robot
            ubxx = [self.ubsx,self.ubsy]*n_robot
        else:
            # Bounds of U
            lbu = [-np.inf, -np.inf]
            ubu = [ np.inf,  np.inf]
            # Bounds on X
            lbxx = [-np.inf, -np.inf]*n_robot
            ubxx = [ np.inf,  np.inf]*n_robot
        # concatenating X and U bounds
        lbx = np.array((lbu + lbxx)*steps*n_mode)
        ubx = np.array((ubu + ubxx)*steps*n_mode)
        # concatenating optimization parameter
        p = ca.reshape(BC, -1, 1)
        return optim_var, lbx, ubx, p

    def _get_objective(self):
        """Returns objective function for optimization.
        If sparse = True, then it returns first norm objective function
        that favors sparsity.
        """
        steps = self.steps
        n_mode = self.specs.n_mode
        U = self.U
        obj = 0
        for step in range(0,steps):
            for mode in range(n_mode):
                i = step*n_mode + mode
                if i >0:
                    u = U[:,i]
                    obj += ca.sum1(u*u)
        """ for i in range(U.shape[1]):
            u = U[:,i]
            obj += ca.sum1(u*u) """
        return obj

    def _get_optimization(self, solver_name = 'ipopt', boundary = False):
        """Sets up and returns a CASADI optimization object."""
        # Construct optimization symbolic vars in CASADI.
        self.X, self.U, self.BC = self._construct_vars()
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

    def _solvers(self):
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
        # Printing level.
        solvers['knitro']['opts']['knitro.outlev'] = 0
        solvers['knitro']['opts']['knitro.outmode'] = 0
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
        solvers['knitro']['opts']['knitro.feastolabs'] = 1
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
        mode_sequence = self.mode_sequence
        mode_start = mode_sequence[np.nonzero(mode_sequence)[0][0]]
        steps = self.steps
        n_robot = self.specs.n_robot
        n_mode = self.specs.n_mode
        xi = self.xi
        U_raw = ca.reshape(sol['x'],2+2*n_robot,-1)[:2,:].full()
        X_raw = ca.horzcat(xi,ca.reshape(sol['x'],2+2*n_robot,-1)[2:,:]).full()
        UZ = np.zeros((3,2*n_mode*steps))
        X = np.zeros((2*n_robot, 2*n_mode*steps + 1))
        X[:,0] = xi
        U = np.zeros_like(UZ)
        # Adjust mode_change parameters.
        dir_mode = 1 if mode_start%2 else -1
        mode_change = np.array([self.specs.mode_rel_length[1], 0.0])*dir_mode
        mode_change_remainder  = ((n_mode-1)%2)*mode_change
        # Recovering input with transitions
        counter = 0
        for step in range(steps):
            for mode in range(0,n_mode):
                # Map the mode to the current swarm mode sequence.
                mode_mapped = mode_sequence[mode]
                if mode_mapped == 0:
                    # This tumbling.
                    U_tumbled = self._accurate_rotation(
                           U_raw[:,step*n_mode + mode] - mode_change_remainder)
                    UZ[:2,counter:counter+2] = U_tumbled
                    UZ[2,counter:counter+2] = mode_mapped
                else:
                    # This is pivot_walking and needs mode change.
                    UZ[:2,counter] = U_raw[:,step*n_mode + mode]
                    UZ[2,counter] = mode_mapped
                    # Calculate next mode and perform it.
                    next_mode_mapped = self.next_mode_sequence[mode]
                    UZ[:2,counter+1] = mode_change
                    UZ[2,counter+1] = -next_mode_mapped
                    mode_change *= -1
                # Calculate positions.
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
        return U_raw, X_raw, UZ,  U, X

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

    def solve(self, xi, xf, receding = 0):
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
        receding: Boolean, default: False
            Use receding horizon or open loop mode.
            In receding horizon mode is used. Only first section of the
            plan plus its related mode change will be returned and the
            planner shifts shifts the robot modes to the next one
            in initial given mode sequence.
        """
        lbx, ubx = self.lbx, self.ubx
        lbg, ubg = self.lbg, self.ubg
        solver = self.solver
        self.xi = xi
        self.xf = xf
        U0 = ca.DM.zeros(self.U.shape)
        X0 = ca.DM(np.matlib.repmat(xi,self.X.shape[1],1).T)
        UX0 = ca.vertcat(ca.reshape(U0,-1,1), ca.reshape(X0,-1,1))
        p = np.hstack((xi, xf))
        sol = solver(x0 = UX0, lbx = lbx, ubx = ubx,
                     lbg = lbg, ubg = ubg, p = p)
        UX_raw = sol['x'].full().squeeze()
        g_raw = sol['g'].full().squeeze()
        isfeasible = self._isfeasible(UX_raw,g_raw)
        #sol = solver(x0 = x0, lbg = lbg, ubg = ubg, p = p)
        # recovering the solution in appropriate format
        BC = np.vstack((xi,xf)).T
        U_raw, X_raw, UZ, U, X = self._post_process_u(sol)
        if receding:
            # Update mode sequences.
            self.mode_sequence.rotate(-receding)
            self.next_mode_sequence.rotate(-receding)
            # Rebuild the optimization problem.
            self._get_optimization(self.solver_name, self.boundary)
        return sol, U_raw, X_raw, BC, UZ, U, X, isfeasible
    
    @classmethod
    def plan(cls,xg,outer_steps,mode_sequence,specs,boundary):
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
                                solver_name='knitro', boundary=boundary)
        # Get new initial condition.
        xi = yield None
        _, _, _, _, _, U, _, isfeasible= planner.solve(xi,xg)
        print(f"{isfeasible = } at {outer_steps:01d} outer_steps.")
        if not isfeasible:
            raise RuntimeError
        yield U.T
        return planner

def receding_example():
    threshold = 3.0
    a, ap = [0,0], [0,20]
    b, bp = [20,0], [20,20]
    c, cp = [40,0], [40,20]
    d, dp = [60,0], [60,20]
    e = [75,0]
    xi = np.array(a+b+c)
    xf = np.array([0,30]+[0,50]+[25,50])
    #xf = np.array(dp+cp+bp+ap)
    outer = 3
    boundary = True
    last_section = False
    pivot_length = np.array([[10,9,8],[9,8,10]])
    mode_sequence= [1,2,0]
    specs=model.SwarmSpecs(pivot_length,10)
    #specs = model.SwarmSpecs.robo(3)
    swarm = model.Swarm(xi, 0, 1, specs)
    planner = Planner(specs, mode_sequence = mode_sequence, steps = outer,
                            solver_name='knitro', boundary=boundary)
    receding = 3
    repetitions = 10#np.ceil(outer*swarm.specs.n_mode/receding).astype(int)
    for i in range(repetitions):
        sol, U_raw, X_raw, P, UZ, U, X, isfeasible = planner.solve(xi, xf,receding)
        print(i)
        print(isfeasible)
        swarm.simplot(U[:,:2*receding],1000, boundary=boundary,
                                             last_section=last_section)
        if i < repetitions - 1:
            noise = np.random.uniform(-2,2,2*swarm.specs.n_robot)
            xi = swarm.position + 1*noise
            swarm.position = xi
            print(noise)
        if np.linalg.norm(xf-np.tile(xf[:2],specs.n_robot)
                        - xi+np.tile(xi[:2],specs.n_robot), np.inf) < threshold:
            break
    if np.linalg.norm(xf[:2] - xi[:2], np.inf) > threshold:
        noise = np.random.uniform(-2,2,2*swarm.specs.n_robot)
        xi = swarm.position + 0*noise
        swarm.position = xi
        print(noise)
        U = np.zeros((3,2))
        print((xf[:2] - xi[:2]))
        U[:2,:] = planner._accurate_rotation(xf[:2] - xi[:2])
        # Convert to polar
        for i in range(U.shape[1]):
            U[:2,i] = planner._cartesian_to_polar(U[:2,i])
            if U[2,i] == 0:
                # If this is rotation round it for numerical purposes.
                U[0,i] = int(U[0,i]*1000)/1000
        swarm.simplot(U,1000,boundary=boundary,last_section=last_section)
        print(swarm.position)
    
########## test section ################################################
if __name__ == '__main__':
    a, ap = [0,0], [0,20]
    b, bp = [20,0], [20,20]
    c, cp = [40,0], [40,20]
    d, dp = [60,0], [60,20]
    e = [75,0]
    xi = np.array(a+b+c)
    transfer = np.array([0,0]*(len(xi)//2))
    xi = xi + transfer
    A = np.array([-15,0]+[-15,30]+[0,45]+[15,30]+ [15,0])
    F = np.array([0,0]+[0,30]+[0,50]+[25,50]+ [20,30])
    M = np.array([-30,0]+[-15,60]+[0,40]+[15,60]+ [30,0])
    xf = F
    #xf = np.array([0,0]+[0,50]+[25,50]+ [20,30])
    xf = np.array([0,30]+[0,50]+[25,50])
    #xf = np.array(dp+cp+bp+ap)
    outer = 3
    boundary = True
    last_section = False

    pivot_length = np.array([[10,9,8],[9,8,10]])
    #pivot_length = np.array([[10,9,8,7],[9,8,7,10],[8,7,10,9],[7,10,9,8]])
    #pivot_length = np.array([[10,9,8,7,6],[9,8,7,6,10],[8,7,6,10,9],[7,6,10,9,8]])

    mode_sequence= [1,2,0]
    specs=model.SwarmSpecs(pivot_length,10)
    specs = model.SwarmSpecs.robo(3)
    swarm = model.Swarm(xi, 0, 1, specs)
    #planner = Planner(specs, mode_sequence , steps = outer,
    #                        solver_name='knitro', boundary=boundary)
    #g, lbg, ubg = planner._get_constraints()
    #optim_var, lbx, ubx, p = planner._get_optim_vars(boundary)
    #obj = planner._get_objective()
    #solver = planner._get_optimization(solver_name='knitro', boundary=boundary)
    #sol, U_raw, X_raw, P, UZ, U, X, isfeasible = planner.solve(xi, xf)
    #print(isfeasible)
    planner = Planner.plan(xf,outer,mode_sequence,specs,boundary)
    U = next(planner)
    if U is None:
        U = planner.send(xi)
        U = U.T
    swarm.reset_state(xi,0,1)
    #anim =swarm.simanimation(U,1000,boundary=boundary,last_section=last_section,save = False)
    swarm.reset_state(xi,0,1)
    swarm.simplot(U,1000,boundary=boundary,last_section=last_section)
