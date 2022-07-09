#%%
########################################################################
# This files hold classes and functions that controls swarm of 
# milirobot system with feedback from camera.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from collections import deque
from math import remainder

import numpy as np
from scipy.spatial.transform import Rotation

# from "foldername" import filename, this is for ROS compatibility.
try:
    from swarm import model, planner
except ModuleNotFoundError:
    import model, planner

np.set_printoptions(precision=2, suppress=True)
########## classes and functions #######################################
class Controller():
    """
    This class holds feedback planning and control for swarm of 
    millirobots. For more info on the algorithms refer to the paper.
    """
    def __init__(self, specs: model.SwarmSpecs,
                       pos: np.ndarray, theta: float, mode: int):
        self.specs = specs
        self.theta_inc = specs.theta_inc
        self.alpha_inc = specs.alpha_inc
        self.rot_inc = specs.rot_inc
        self.pivot_inc = specs.pivot_inc
        self.tumble_inc = specs.tumble_inc
        self.theta_sweep = specs.theta_sweep # Max theta sweep.
        self.alpha_sweep = specs.alpha_sweep # Max alpha sweep.
        self._set_rotation_constants_and_functions()
        self.reset_state(pos, theta, 0, mode)
        self.power = 100.0

    def _set_rotation_constants_and_functions(self):
        """
        This function initializes magnets vectors for each mode.
        It also constructs lambda functions for rotating 
        given vectors for given amounnts around certain global axes.
        """
        # Set up rotation functions.
        self.rotx = lambda vect, ang: Rotation.from_euler('x', ang).apply(vect)
        self.roty = lambda vect, ang: Rotation.from_euler('y', ang).apply(vect)
        self.rotz = lambda vect, ang: Rotation.from_euler('z', ang).apply(vect)
        # Rotation around a given axis, angles based on right hand rule.
        self.rotv = (lambda vect, axis, ang: 
                          Rotation.from_rotvec(ang*axis).apply(vect).squeeze())
        # Set magnet vector
        self.magnet_vect = np.array([0.0,-1.0,0.0])

    def reset_state(self, pos: np.ndarray= None, theta= None,
                          alpha= None, mode: int =None):                        
        # Process input.
        pos = self.pos if pos is None else np.array(pos)
        theta = self.theta if theta is None else self.wrap(theta)
        alpha = self.alpha if alpha is None else alpha
        mode = self.mode if mode is None else int(mode)
        if (pos.shape[0]//2 != self.specs.n_robot):
            msg = "Position does not match number of the robots."
            raise ValueError(msg)
        #
        pivot_length = self.specs.pivot_length[mode,:]/2
        self.pos = pos.astype(float)
        # Calculate leg positions.
        self.posa = np.zeros_like(self.pos)
        self.posb = np.zeros_like(self.pos)
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        for robot in range(self.specs.n_robot):
            # a is half way along +y or leg_vect.
            self.posa[2*robot:2*robot+2] = (self.pos[2*robot:2*robot+2]
                                           +pivot_length[robot]*leg_vect)
            # b is half way along -y or -leg_vect
            self.posb[2*robot:2*robot+2] = (self.pos[2*robot:2*robot+2]
                                           -pivot_length[robot]*leg_vect)
        self.theta = theta
        self.alpha = alpha
        self.mode = mode
        self.update_mode_sequence(mode)

    def update_mode_sequence(self,mode: int):
        self.mode_sequence = deque(range(1,self.specs.n_mode))
        self.mode_sequence.rotate(-mode+1)
    
    def get_state(self):
        return self.pos,self.theta,self.alpha,self.mode,self.posa,self.posb

    def body2magnet(self, ang):
        """
        Converts (theta, alpha) of robots body, to (theta, alpha) of 
        the robots magnets, to use as desired orientation of the field.
        ----------
        Parameters
        ----------
        ang: array = [theta, alpha] radians.
        ----------
        Returns
        ----------
        magnet: np.ndarray = [theta_m, alpha_m] degrees
        """
        # Calculate the magnet vetor in cartesian coordinate.
        magnet = self.rotx(self.magnet_vect, ang[1]) # Alpha rotation.
        magnet = self.rotz(magnet, ang[0])           # Theta rotation.
        # Convert to spherical coordinate, [theta_m, alpha_m].
        magnet = self.cart_to_sph(magnet)
        return magnet
    
    @staticmethod
    def cart_to_sph(cartesian):
        """Converts cartesian vector to spherical degrees."""
        # alpha: arctan(z/(x**2 + y**2)**.5)
        alpha = np.degrees(np.arctan2(cartesian[2],
                                      np.linalg.norm(cartesian[:2])))
        # theta: arctan(y/x)
        theta = np.degrees(np.arctan2(cartesian[1], cartesian[0]))
        return [theta, alpha] # Should be list.
    
    @staticmethod
    def cart2pol(cart):
        return [np.linalg.norm(cart), np.arctan2(cart[1],cart[0])]

    @staticmethod
    def pol2cart(pol):
        return [pol[0]*np.cos(pol[1]), pol[0]*np.sin(pol[1])]
    
    @staticmethod
    def frange(start, stop=None, step=None):
        """Float version of python range"""
        # This function is from stackoverflow.
        # if set start=0.0 and step = 1.0 if not specified
        start = float(start)
        if stop == None:
            stop = start + 0.0
            start = 0.0
        if step == None:
            step = 1.0
        if step == 0.0:
            raise ValueError("frange() arg 3 must not be zero.")
        #
        count = 0
        while True:
            temp = float(start + count * step)
            if step > 0 and temp >= stop:
                break
            elif step < 0 and temp <= stop:
                break
            yield temp
            count += 1

    @staticmethod
    def wrap(angle):
        """Wraps angles between -PI to PI."""
        wrapped = remainder(-angle+np.pi, 2*np.pi)
        if wrapped< 0:
            wrapped += 2*np.pi
        wrapped -= np.pi
        return -wrapped
    
    @staticmethod
    def wrap_range(from_ang, to_ang, inc = 1):
        """
        Yields a range of wrapped angle that goes from \"from_ang\"
        to \"to_ang\".
        """
        diff = Controller.wrap(to_ang - from_ang)
        inc = -inc if diff <0 else inc
        to_ang = from_ang + diff
        for ang in Controller.frange(from_ang, to_ang, inc):
            yield Controller.wrap(ang)
    
    @staticmethod
    def range_oval(theta_start, theta_end, sweep_alpha, inc):
        """
        This function produces a range tuple for theta and alpha.
        The angles are produced so that the lifted end of the robot will
        go on an half oval path. The oval majpr axes lengths are as:
        horizontal ~ tan(sweep_theta)
        vertical   ~ tan(sweep_alpha)
        """
        sweep_theta = Controller.wrap(theta_end - theta_start)/2
        theta_sgn = -1 if sweep_theta < 0 else 1
        alpha_sgn = -1 if sweep_alpha < 0 else 1
        sweep_theta = abs(sweep_theta)
        sweep_alpha = abs(sweep_alpha)
        msg = "Acceptable ranges: 0 < arg3 =< pi/3, agr4 > 0"
        assert sweep_alpha > 0 and sweep_alpha <= np.pi/2 and inc > 0, msg
        # r = f(ang), oval polar coordinate equation.
        fr = lambda ang: (np.tan(sweep_alpha)*np.tan(sweep_theta)
                         /np.sqrt((np.tan(sweep_alpha)*np.cos(ang))**2
                                 +(np.tan(sweep_theta)*np.sin(ang))**2))
        #
        for ang in Controller.frange(0,np.pi,inc):
            # Get corresponding radius on the ellipse.
            r = fr(ang)
            # Calculate angles and yield them.
            alpha = np.arctan2(r*np.sin(ang), 1)
            theta = sweep_theta - np.arctan2(r*np.cos(ang), 1)
            yield Controller.wrap(theta_start+theta_sgn*theta),alpha_sgn*alpha
        # Yield last one.
        yield Controller.wrap(theta_end), 0.0
        # Repeat last, for more delay and accuracy.
        yield Controller.wrap(theta_end), 0.0

    def step_alpha(self, desired:float, inc = None):
        """
        Yields body angles that transitions robot to desired alpha.
        """
        inc = self.alpha_inc if inc is None else inc
        initial = self.alpha
        for alpha in self.wrap_range(initial, desired, inc):
            self.update_alpha(alpha)
            yield [self.theta, alpha]
        # Yield last section.
        alpha = self.wrap(desired)
        self.update_alpha(alpha)
        yield [self.theta, alpha]
        # Repeat last, for more delay and accuracy.
        if abs(alpha) < 1:
            yield [self.theta, alpha]
    
    def update_alpha(self, alpha: float):
        self.alpha = self.wrap(alpha)

    def step_theta(self, desired:float, pivot: str = None):
        """
        Yields body angles that transitions robot to desired theta.
        """
        initial = self.theta
        inc = self.rot_inc if pivot is None else self.theta_inc
        for theta in self.wrap_range(initial, desired, inc):
            self.update_theta(theta, pivot)
            yield [theta, self.alpha]
        # Yield last section.
        theta = self.wrap(desired)
        self.update_theta(theta, pivot)
        yield [theta, self.alpha]
        # Repeat last, for more delay and accuracy.
        if abs(self.alpha) < 1:
            yield [theta, self.alpha]
    
    def update_theta(self, theta:float, pivot: str = None):
        theta = self.wrap(theta)
        pivot_length = self.specs.pivot_length[self.mode,:]
        # Leg vector, along +y (robot body frame) axis.
        leg_vect = self.rotz(np.array([0,1.0,0]), theta)[:2]
        # Update positions of all robots.
        if pivot == 'a':
            # posa is intact.
            for robot in range(self.specs.n_robot):
                # pos is @ -leg_vect/2.
                self.pos[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                              -pivot_length[robot]*leg_vect/2)
                # posb is @ -leg_vect.
                self.posb[2*robot:2*robot+2] = (self.posa[2*robot:2*robot+2]
                                             -pivot_length[robot]*leg_vect)
        elif pivot == "b":
            # posb remains intact.
            for robot in range(self.specs.n_robot):
                # pos is @ +leg_vect/2.
                self.pos[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                              +pivot_length[robot]*leg_vect/2)
                # posa is @ +leg_vect.
                self.posa[2*robot:2*robot+2] = (self.posb[2*robot:2*robot+2]
                                               +pivot_length[robot]*leg_vect)
        else:
            # Rotationg around center.
            assert abs(self.alpha) <.01, "Robots is lifted."
            # pos remains intact, call reset_state.
            self.reset_state(theta = theta)
        # Update theta
        self.theta = theta
    
    def get_pivot(self,phi, theta):
        """Determines whick leg should be used as pivot."""
        # Determine pivot to avoid unnecessary rotation.
        delta = self.wrap(theta - phi)
        dir_pivot = -1 if delta > 0 else 1 # -1 is B, +1 is A.
        dir_ang = -1 if (delta > np.pi/2 or delta <= -np.pi/2) else 1
        cte = np.pi if (delta > np.pi/2 or delta <= -np.pi/2) else 0
        return dir_pivot, dir_ang, cte

    def step_mode(self, des_mode: int, phi, last= False, line_up= True):
        """
        Yields body angles that transitions robot to desired mode at
        the given angle.
        ----------
        Parameters
        ----------
        des_mode: int: Desired mode to go.
        phi: Direction of movement in the mode change w.r. robot center.
        line_up: Bool: line_up robot before mode change.
        """
        str_msg = "Mode change can only be started from alpha = 0."
        assert abs(self.alpha) <.01, str_msg
        # Determine pivot to avoid unnecessary rotation.
        dir_pivot, dir_ang, cte = self.get_pivot(phi, self.theta)
        # Get index of desired mode from the current sequence to
        # determine parameters for tumbling process.
        # Mode index to get tumbling angles and distance.
        if self.mode != des_mode:
            des_mode_index = self.mode_sequence.index(des_mode)
            rel_ang = self.specs.mode_rel_ang[des_mode_index]
            if line_up:
                theta_start = phi + cte - dir_ang*dir_pivot*rel_ang/2
                theta_end = phi + cte + dir_ang*dir_pivot*rel_ang/2
            else:
                theta_start = self.theta
                phi = theta_start + rel_ang/2
                theta_end = self.theta + rel_ang
                dir_pivot = 1
            # Positions 
            pos_start = self.pos
            pos_delta = (self.specs.mode_rel_length[des_mode_index]*
                         np.array([np.cos(phi), np.sin(phi)]*
                         self.specs.n_robot))
            pos_end = pos_start + pos_delta
            # Line up the robots based on the mode change angle.
            yield from self.step_theta(theta_start)
            # Lift the robots.
            yield from self.step_alpha(-dir_pivot*np.pi/2,self.tumble_inc)
            # Update theta and mode, and position.
            self.reset_state(pos = pos_end, theta=theta_end, mode=des_mode)
            # Lift down the robots.
            yield from self.step_alpha(0.0, self.tumble_inc)
            if last:
                # Line up the robot.
                yield from self.step_theta(phi + cte)
        else:
            yield from self.step_theta(self.theta)

    def mode_changing(self, input_cmd, last= False, line_up= True):
        """
        High level command to change mode.
        @param: array as [_, phi, mode_to_go]
        """
        des_mode = int(abs(input_cmd[2]))
        phi = input_cmd[1]
        yield from map(self.body2magnet, self.step_mode(des_mode, phi,
                                                        last, line_up))
    
    def step_tumble(self, phi, steps: int, last= False,line_up= True):
        """
        Yields magnet angles that walks the robot using tumbling mode.
        """
        assert abs(self.alpha) <0.01, "Tumbling should happen at alpha = 0."
        # Determine pivot to avoid unnecessary rotation.
        dir_pivot, dir_ang, cte = self.get_pivot(phi, self.theta)
        theta_start = phi + cte - dir_ang*dir_pivot*np.pi/2
        # Calculate position update step. Only valid when lined_up.
        pos_delta = (self.specs.tumbling_length
                    *np.array([np.cos(phi), np.sin(phi)]*self.specs.n_robot))
        # Line up robots. Can disable only when calling consequtively.
        if line_up:
            yield from self.step_theta(theta_start)
        # Set tumbling parameter.
        next_mode={0:self.mode_sequence[(self.specs.n_mode-1)//2],1:self.mode}
        # Perform tumbling.
        for step in range(steps):
            # Lift the robot.
            yield from self.step_alpha(-dir_pivot*np.pi/2, self.tumble_inc)
            # Update position, theta, and mode
            self.reset_state(pos = self.pos + pos_delta,
                             theta = self.theta + np.pi,
                             mode = next_mode[step%2])
            # Take down the robot.
            yield from self.step_alpha(0, self.tumble_inc)
            # Update alpha_direction.
            dir_pivot *= -1
        # Line up the robots, if this was last section.
        if last is True:
            yield from self.step_theta(phi + cte)

    def tumbling(self, input_cmd, last = False, line_up = True):
        """
        High level function for step_tumble.
        @param: array as [distance to walk, theta, mode]
        """
        # determine rounded number of rotations needed.
        steps = round(input_cmd[0]/self.specs.tumbling_length)
        yield from map(self.body2magnet,
                       self.step_tumble(input_cmd[1], steps, last, line_up))

    def rotation(self, input_cmd, minimal = False):
        """Rotates robots in place."""
        if minimal:
            _, _, cte = self.get_pivot(input_cmd[1], self.theta)
            input_cmd[1] = self.wrap(input_cmd[1] + cte)
        yield from map(self.body2magnet,self.step_theta(input_cmd[1]))

    def step_pivot(self, phi, sweep, steps: int, last= False, line_up= True):
        """
        Yields field angles for pivot walking with specified sweep angles
        and number of steps.
        """
        assert steps >= 0, "\"steps\" should be positive integer."
        phi = self.wrap(phi)
        pivot = {1:"a", -1:"b"}
        # Determine pivot to avoid unnecessary rotation.
        dir_pivot, dir_ang, cte = self.get_pivot(phi, self.theta)
        # Line of the robot for currect direction.
        if line_up:
            yield from self.step_theta(phi + cte - dir_ang*dir_pivot*sweep)
        for _ in range(steps):
            theta_start = self.theta
            theta_end = phi + cte + dir_ang*dir_pivot*sweep
            for theta, alpha in Controller.range_oval(theta_start, theta_end,
                                                   -dir_pivot*self.alpha_sweep,
                                                    self.pivot_inc):
                # Update relted states.
                self.update_alpha(alpha)
                self.update_theta(theta, pivot[dir_pivot])
                # Yield the values.
                yield [self.theta, self.alpha]
            # Update pivot.
            dir_pivot *= -1
        if last:
            # Line up the robot.
            yield from self.step_theta(phi + cte)

    def pivot_walking(self, phi, sweep, steps:int, last= False, line_up= True):
        """Wrapper arounf step_pivot."""
        yield from map(self.body2magnet,
                       self.step_pivot(phi, sweep, steps, last, line_up))
    
    def get_pivot_params(self, r, phi):
        """
        Calculates number of steps and sweep angle for pivot walking.
        """
        # Get pivot length of leader robot in current mode.
        pivot_length = self.specs.pivot_length[self.mode,0]
        # Calculate maximum distance the leader can travel in one step.
        d_step_max = pivot_length*np.sin(self.theta_sweep)
        # Number of steps takes to do the pivot walk.
        steps = int(np.ceil(r/d_step_max))
        d_step = r/steps if steps else 0
        sweep = np.arcsin(d_step/pivot_length) if steps else 0
        return steps, sweep

    def pivot_walking_walk(self,input_cmd, last = False, line_up = True):
        """
        Calls pivot walking and yields field angles and states.
        @param: Numpy array as [n_steps, starting_theta (degrees)]
        """
        assert input_cmd[0] >=0, "n_steps cannot be negative."
        steps = int(input_cmd[0])
        phi = input_cmd[1]
        self.reset_state(theta = phi)
        if not line_up: # Adjust for taking full step in first step.
            phi = self.wrap(phi + self.theta_sweep)
            line_up = True
        yield from self.pivot_walking(phi,self.theta_sweep,steps,last,line_up)

    def feedforward_pivot(self, input_cmd, last= False):
        """
        Generates and yields body angles for pivot walking.
        @param: Numpy array as [distance to walk, theta, mode]
        """
        # Check if the commanded input mode matched the current mode.
        if input_cmd[2] != self.mode:
            exc_msg = "Input is incompatible with current mode."
            raise ValueError(exc_msg)
        # Determine walking parameters.
        steps, sweep = self.get_pivot_params(input_cmd[0],input_cmd[1])
        yield from self.pivot_walking(input_cmd[1],sweep,steps,
                                                             last,line_up=True)
    
    def feedforward_line(self, input_series, has_last= True):
        """
        Yields magnetic field for executing series of input feedforward.
        @param: 2D array of commands as [distance, angle, mode] per row.
        """
        num_sections = len(input_series)
        for section in range(num_sections):
            last = False if (section < num_sections-1) else has_last
            input_cmd = input_series[section]
            input_mode = int(input_cmd[2])
            if input_mode < 0:
                # This is mode change request.
                yield from self.mode_changing(input_cmd, last)
            elif input_mode == 0:
                # This is tumbling.
                yield from self.tumbling(input_cmd, last)
            elif input_mode == 999:
                # This is rotation in place.
                yield from self.rotation(input_cmd)
            else: 
                # This is pivot walking mode.
                yield from self.feedforward_pivot(input_cmd,last)

    # Closed loop related functions.
    def modify_angles(self,angles,angle):
        """
        Modifies recorded angle of robots based on the given angle.
        Recorded angle is the minimum angle of robot w.r.t. x axis ccw
        and it is +-pi/2 range.
        ----------
        Parameters
        ----------
        angles: array of radians.
        angle: scalar
        ----------
        Returns
        ----------
        angles: array, modified angles.
        """
        for i, ang in enumerate(angles):
            a = [ang, self.wrap(ang + np.pi)]
            idx = np.argmin([abs(self.wrap(a[0]-angle)),
                             abs(self.wrap(a[1]-angle))])
            angles[i] = a[idx]
        return angles
            
    def process_robots(self, robot_states, any_robot = True):
        """
        Returns pos and theta angle of robots found.
        ----------
        Parameters
        ----------
        robot_states: array [xi, yi, theta_i, ...]
        any_robot: Boolean, Default: True
            If true, it only return states for the first robot found.
        ----------
        Exception
        ----------
        ValueError, if no robot found.
        ----------
        Returns
        ----------
        position: np.array [x_i,y_i, ...]
        angle: theta
        """
        n_robot = self.specs.n_robot
        positions, angles = [], []
        robot_states = np.array(robot_states,dtype=float)
        robot_states = np.reshape(robot_states,(-1,3))
        if any_robot:
            # Returns the first robot that it found.
            for state in robot_states:
                if 999 not in state:
                    positions.extend(state[:2])
                    angles.append(state[2])
                    break
            positions *= n_robot
            angles *= n_robot
            if not positions:
                raise ValueError("No robots found.")
        else:
            # Looks for all robots available in current robot specs.
            for robot in range(n_robot):
                state = robot_states[robot]
                if 999 in state:
                    raise ValueError("One or more robot is missing.")
                positions.extend(state[:2])
                angles.append(state[2])
        # Fix angle to closest one.
        angles = self.modify_angles(angles,self.theta)
        return np.array(positions), angles[0]
    
    def closed_pivot_cart(self, xg, last = False, average = False):
        """
        Generates and yields body angles for closed loop pivot walking.
        @param: Numpy array as [distance to walk, theta, mode]
        """
        line_up = True
        B = self.specs.B[self.mode]
        cnt = 0
        xg = np.array(xg,dtype=float)
        xf = xg
        while True:
            state_fb = yield None
            xi, _ = state_fb
            self.reset_state(pos = xi)
            # Calculate equivalent polar command.
            if average:
                xf = xi + np.dot(np.dot(B,np.linalg.pinv(B)),xg - xi)
            input_cmd = self.cart2pol(xf[:2] - xi[:2])
            # Determine walking parameters.
            steps, sweep = self.get_pivot_params(input_cmd[0],input_cmd[1])
            yield from self.pivot_walking(input_cmd[1],sweep,1,False,line_up)
            if steps < 2: break
            line_up = False
            cnt += 1
    
    def pivot_calibration_walk(self, input_cmd):
        """feedforfard walking for calibraiton purpose."""
        r, phi = input_cmd[0], input_cmd[1]
        # Align the robot.
        yield from self.pivot_walking(phi,self.theta_sweep,0)
        yield from self.pivot_walking(phi,self.theta_sweep,0) # Delay
        state_fb = yield None
        xi, _ = state_fb
        self.reset_state(pos = xi)
        xs = self.get_state()[0] # Initial position.
        yield from self.pivot_walking(phi,self.theta_sweep,0)
        # Pivot walk until at least r is travelled.
        cnt = 0
        while True:
            state_fb = yield None
            xc, _ = state_fb
            rc, ang = self.cart2pol(xc[:2] - xs[:2])
            yield from self.pivot_walking(phi,self.theta_sweep,1,line_up=False)
            if rc > r:
                return 2*rc/cnt, self.wrap(ang - phi)
            cnt += 1

    def pivot_calibration(self, n_sect):
        """Yields field for pivot walking calibration prcess."""
        lengths = []
        ang_errs = []
        r = 70
        d = 2*r
        angles = np.linspace(0,np.pi, int(n_sect) + 1)[:-1]
        for ang in angles:
            theta_s = self.wrap(ang + np.pi)
            theta_e = ang
            xs = self.pol2cart([r, ang])*self.specs.n_robot
            # Go to starting point.
            yield from self.closed_pivot_cart(xs)
            # Forward pass.
            length,ang_err= yield from self.pivot_calibration_walk([d,theta_s])
            lengths.append(length)
            ang_errs.append(ang_err)
            #Backward pass.
            length,ang_err= yield from self.pivot_calibration_walk([d,theta_e])
            lengths.append(length)
            ang_errs.append(ang_err)
        lengths = np.array(lengths).reshape(-1,2)
        ang_errs = np.array(ang_errs).reshape(-1,2)
        # Print
        msg = "Calibration stats:\n"
        msg+= "\n".join(
            (f"ang: {np.rad2deg(ang):+07.2f}, "
            +f"length: {piv[0]:+07.2f}, {piv[1]:+07.2f}, "
            +f"ang_err: {np.rad2deg(ang_err[0]):+07.2f}, "
            +f"{np.rad2deg(ang_err[1]):+07.2f}")
            for ang, piv, ang_err in zip(angles,lengths, ang_errs))
        msg += f"\nAverage pivot length:{np.mean(lengths):+07.2f}"
        msg+= f"\nCoeff of variation:{np.std(lengths)/np.mean(lengths):+07.2f}"
        msg += f"\nAverage ang_err:{np.rad2deg(np.mean(ang_errs)):+07.2f}"
        return msg

    def tumble_calibration_walk(self, input_cmd):
        """feedforfard walking for calibraiton purpose."""
        r, phi = input_cmd[0], input_cmd[1]
        one = self.specs.tumbling_length
        # Align the robot.
        yield from self.tumbling([0,phi])
        yield from self.tumbling([0,phi]) # Delay
        state_fb = yield None
        xi, _ = state_fb
        self.reset_state(pos = xi)
        xs = self.get_state()[0] # Initial position.
        yield from self.tumbling([0,phi])
        # Pivot walk until at least r is travelled.
        cnt = 0
        while True:
            state_fb = yield None
            xc, _ = state_fb
            rc, ang = self.cart2pol(xc[:2] - xs[:2])
            yield from self.tumbling([one,phi], line_up=False)
            if rc > r:
                return rc/cnt, self.wrap(ang - phi)
            cnt += 1
    
    def tumble_calibration(self, n_sect):
        """Yields field for pivot walking calibration prcess."""
        lengths = []
        ang_errs = []
        r = 65
        d = 2*r
        angles = np.linspace(0,np.pi, int(n_sect) + 1)[:-1]
        for ang in angles:
            theta_s = self.wrap(ang + np.pi)
            theta_e = ang
            xs = self.pol2cart([r, ang])*self.specs.n_robot
            # Go to starting point.
            yield from self.closed_pivot_cart(xs)
            # Forward pass.
            length,ang_err=yield from self.tumble_calibration_walk([d,theta_s])
            lengths.append(length)
            ang_errs.append(ang_err)
            #Backward pass.
            length,ang_err=yield from self.tumble_calibration_walk([d,theta_e])
            lengths.append(length)
            ang_errs.append(ang_err)
        lengths = np.array(lengths).reshape(-1,2)
        # Print
        lengths = np.array(lengths).reshape(-1,2)
        ang_errs = np.array(ang_errs).reshape(-1,2)
        # Print
        msg = "Calibration stats:\n"
        msg+= "\n".join(
            (f"ang: {np.rad2deg(ang):+07.2f}, "
            +f"length: {piv[0]:+07.2f}, {piv[1]:+07.2f}, "
            +f"ang_err: {np.rad2deg(ang_err[0]):+07.2f}, "
            +f"{np.rad2deg(ang_err[1]):+07.2f}")
            for ang, piv, ang_err in zip(angles,lengths, ang_errs))
        msg += f"\nAverage length per tumble:{np.mean(lengths):+07.2f}"
        msg+= f"\nCoeff of variation:{np.std(lengths)/np.mean(lengths):+07.2f}"
        msg += f"\nAverage ang_err:{np.rad2deg(np.mean(ang_errs)):+07.2f}"
        return msg

    def get_cart_goal(self, pos, mode_start, input_cmd):
        """
        This implements input_cmd to the robots at pos and returns the
        corresponding pos after applying each input_cmd.
        This function can also be used to check compatibility of input.
        ----------
        Parameters
        ----------
        pos: array corrent pos of robots.
        mode: int, starting mode.
        input_cmd: 2D array as [r, phi, mode]
           r:     Distance to travel, irrelevant if mode < 0.
           phi:   Angle to travel
           mode: int, the mode motion is performed.
        ----------
        Returns
        ----------
        poses: 2D array of pos after each input_cmd.
        """
        n_mode = self.specs.n_mode
        mode_sequence = deque(range(1,self.specs.n_mode))
        mode_sequence.rotate(-mode_start+1)
        pos = np.array(pos)
        input_cmd = np.array(input_cmd).reshape(-1,3)
        poses = []
        # Calculate poses
        for idx, (r, phi, mode) in enumerate(input_cmd):
            mode = int(mode)
            if mode < 0:
                # Mode change request.
                rel_mode_index = mode_sequence.index(-mode)
                r = self.specs.mode_rel_length[rel_mode_index]
                B = self.specs.B[0,:,:]
                mode_sequence.rotate(-rel_mode_index)
            elif mode == 0:
                # Tumbling request.
                rotations = round(r/self.specs.tumbling_length)
                r = rotations*self.specs.tumbling_length
                B = self.specs.B[mode,:,:]
                # Update mode if needed.
                if rotations%2:
                    mode_sequence.rotate(-(n_mode//2))
            elif mode == 999:
                # Rotation.
                r = 0
                B = self.specs.B[0,:,:]
            else:
                # Pivot walking
                if mode != mode_sequence[0]:
                    raise ValueError(f"Incompatible at section: {idx+1:02d}")
                B = self.specs.B[mode,:,:]
            u = [r*np.cos(phi), r*np.sin(phi)]
            pos = pos + np.dot(B, u)
            poses.append(pos)
        return np.array(poses)
    
    def closed_line_single(self,input_cmd, xf,average=False,last=False):
        """
        Executes polar command in closed loop mode.
        ----------
        Parameters
        ----------
        input_cmd: 1D array of polar commands as [r, phi, mode]
        xf: 1D array of cartesian goal.
        average: Bool, default=False, see closed_pivot_cart doc.
        last: Bool, default= False, lines up robots in the end.
        ----------
        Yields
        ----------
        field: magnetic field needed to perform the command.
        """
        #
        _, _, mode = input_cmd
        if mode < 0:
            # Mode change request
            field = yield from self.mode_changing(input_cmd, last)
        elif mode == 0:
            # This is tumbling.
            field = yield from self.tumbling(input_cmd, last)
        elif mode == 999:
            # This is rotation in place.
            field = yield from self.rotation(input_cmd)
        else:
            field = yield from self.closed_pivot_cart(xf, last, average)
        return field

    def closed_line(self, polar_cmd,average=False,has_last=False):
        """
        Executes series of polar command in closed loop mode.
        The get_cart_goal also does compatibility check of the command.
        ----------
        Parameters
        ----------
        polar_cmd: 2D array of polar commands as [r, phi, mode]
        average: Bool, default=False, see closed_pivot_cart doc.
        has_last: Bool, default= False, lines up robots in the end.
        ----------
        Yields
        ----------
        field: magnetic field needed to perform the command.
        """
        tumbling_length = self.specs.tumbling_length
        polar_cmd = np.array(polar_cmd).reshape(-1,3)
        # Remove negligible moves
        polar_cmd = polar_cmd[np.argwhere(polar_cmd[:,0]> 1).squeeze()]
        # Remove unnecessary mode changes.
        mode_change_ind = np.argwhere(polar_cmd[:,2]<0).squeeze()
        index_to_del = np.argwhere(np.diff(mode_change_ind)<2).squeeze()
        index_to_del = np.hstack((index_to_del,index_to_del +1 ))
        index_to_del = mode_change_ind[index_to_del]
        polar_cmd = np.delete(polar_cmd, index_to_del, axis= 0)
        print(polar_cmd)
        goals = self.get_cart_goal(self.pos, self.mode, polar_cmd)
        n_sections = len(polar_cmd)
        ratios = np.zeros((n_sections, 3 + self.specs.n_robot))
        #
        msg = "="*72+"\n"
        msg += "xi:"+",".join(f"{i:+07.2f}" for i in self.pos) + "\n"
        for section, (input_cmd, xg) in enumerate(zip(polar_cmd,goals)):
            #
            msg_1 = f"Section {section+1 } of {n_sections}.\n"
            msg_1 += (f"input:{input_cmd[0]:+07.2f},"
               +f"{np.rad2deg(input_cmd[1]):+07.2f},{int(input_cmd[2]):03d}\n")
            msg_1 += "xg:"+",".join(f"{i:+07.2f}" for i in xg) + "\n"
            print(msg_1,end="")
            last = False if (section < n_sections-1) else has_last
            xi = self.pos
            iterator = self.closed_line_single(input_cmd, xg, average, last)
            for field in iterator:
                if field is None:
                    state_fb = yield field, input_cmd
                    field = iterator.send(state_fb)
                yield field, input_cmd
            xf = self.pos
            eg = xf - xg     # Realized general error.
            d = np.linalg.norm((xf - xi).reshape(-1,2), axis = 1)
            d_ratio = d/d[0] # Ratio of pivot_length w.r.t. leader.
            ratios[section,:-3] = d_ratio
            ratios[section,-3:] = input_cmd
            msg_2 = "xf:"+",".join(f"{i:+07.2f}" for i in xf) + "\n"
            msg_2 += " e:"+",".join(f"{i:+07.2f}" for i in eg) + "\n"
            msg_2 += "ratio:" + ",".join(f"{i:+07.2f}" for i in d_ratio) + "\n"
            msg_2 += "*"*72 + "\n"
            print(msg_2)
            msg += msg_1 + msg_2
        avg_ratio = np.zeros_like(self.specs.beta)
        std_ratio = np.zeros_like(self.specs.beta)
        avg_ratio[0,:] = np.mean(ratios[polar_cmd[:,2]<1],axis=0)[:3]
        std_ratio[0,:] = np.std(ratios[polar_cmd[:,2]<1],axis=0)[:3]
        for i in range(1,self.specs.n_mode):
            avg_ratio[i,:] = np.mean(ratios[polar_cmd[:,2]==i],axis=0)[:3]
            std_ratio[i,:] = np.std(ratios[polar_cmd[:,2]==i],axis=0)[:3]
        stat_ratio = np.hstack((avg_ratio, std_ratio/avg_ratio))
        #ratios = ratios[polar_cmd[:,2].argsort()]
        msg += "ratios:\n"
        msg += "\n".join(",".join(f"{i:+08.3f}" for i in j) for j in ratios)
        msg += "\nratio stats: \n"
        msg +="\n".join(",".join(f"{i:+08.3f}" for i in j) for j in stat_ratio)
        msg += "\nthm_ratios:\n"
        msg += "\n".join(
                   ",".join(f"{i:+08.3f}" for i in j) for j in self.specs.beta)
        msg += "\n"+"="*72
        return msg
    
    def plan_line(self, XG, outer_steps):
        """
        Plans swarm motion for a given goal and executes it.
        ----------
        Parameters
        ----------
        XG: array of final position [x_i, y_i, ...]
        outer_steps: minimum value of outer steps to be used.
        ----------
        Yields
        ----------
        field: magnetic field needed to perform the command.
        input_cmd: array of current input.
        xg: array of current step goal.
        ----------
        Returns
        ----------
        msg: string for showing statistics of the motion.
        """
        # Set up the planner.
        mode_sequence = self.mode_sequence + deque([0])
        plan = planner.Planner.plan(XG,outer_steps,mode_sequence,self.specs)
        for polar_cmd in plan:
            if polar_cmd is None:
                polar_cmd = plan.send(self.pos)
            iterator = self.closed_line(polar_cmd,average=True)
            msg = yield from iterator
        return msg

########## test section ################################################
if __name__ == '__main__':
    specs = model.SwarmSpecs.robo3()
    xi = np.array([0,0,20,0,40,0])
    xf = np.array([10,0,20,0,40,0])
    mode = 1
    control = Controller(specs,xi,0,mode)
    #control.reset_state(theta= np.deg2rad(-30))
    phi = np.deg2rad(135)
    sweep = np.deg2rad(30)
    input_series = np.array([[10,0,1],
                             [0,0,1],
                             [12,0,-2],
                             [12*2,np.pi/2,0],
                             [10,0,2],
                             [12,0,0],
                             [0,np.pi/2,999],
                             [0,np.pi/4,1]])
    iterator = control.step_mode(2, phi, last=False, line_up= True)
    iterator = control.step_tumble(phi, 2, last= False,line_up= True)
    iterator = control.step_pivot(phi, sweep, 10,last= False,line_up= True)
    iterator = control.pivot_walking_walk([2,0], last= False,line_up= True)
    iterator = control.mode_changing([0,phi,mode], last= False, line_up=True)
    iterator = control.tumbling([20,phi],last=False, line_up=True)
    iterator = control.feedforward_pivot([10,phi,1], last= False)
    print(*(i for i in control.get_cart_goal(xi,mode,input_series)), sep="\n")
    iterator = control.feedforward_line(input_series,has_last=True)
    iterator = control.closed_pivot_cart(xf, last = False, average = False)
    try:
        while True:
            field = next(iterator)
            if field is None:
                field = iterator.send((xi,0))
            i = control.get_state()
            #self.pos,self.theta,self.alpha,self.mode,self.posa,self.posb
            str_msg = (",".join(f"{elem:+07.2f}" for elem in i[0]) + "| "
                    +f"{i[3]:01d}| " 
                    +f"{np.rad2deg(i[1]):+07.2f},{np.rad2deg(i[2]):+07.2f}| "
                    +",".join(f"{elem:+07.2f}" for elem in field)
                    )
            print(str_msg)
    except StopIteration as e:
        print(e.value)
        pass
