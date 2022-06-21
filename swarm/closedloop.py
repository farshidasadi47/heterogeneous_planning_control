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
    from swarm import model
except ModuleNotFoundError:
    import model

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
        phi = phi if line_up else self.theta
        dir_pivot, dir_ang, cte = self.get_pivot(phi, self.theta)
        theta_start = phi + cte - dir_ang*dir_pivot*np.pi/2
        # Calculate position update step.
        pos_delta = (self.specs.tumbling_length
                    *np.array([np.cos(phi), np.sin(phi)]*self.specs.n_robot))
        # Line up robots for starting with pivot B.
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

    def rotation(self, input_cmd):
        """Rotates robots in place."""
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
        phi = np.deg2rad(input_cmd[1])
        self.reset_state(theta = phi)
        if not line_up: # Adjust for taking full step in first step.
            phi = self.wrap(phi + np.pi/6)
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
    
    def compatibility_check(self, input_series):
        """
        Checks if an input_series is a compatible sequence.
        @param: 2D array of commands as [distance, angle, mode] per row.
        """
        # Get states, to reset them after compatibility check.
        states = self.get_state()
        input_series = np.array(input_series)
        num_sections = input_series.shape[0]
        try: 
            for section in range(num_sections):
                input_cmd = input_series[section,:]
                input_mode = int(input_cmd[2])
                if input_mode < 0:
                    # This is mode change request.
                    for _ in self.mode_changing(input_cmd, False): pass
                elif input_mode == 0:
                    # This is tumbling.
                    for _ in  self.tumbling(input_cmd, False): pass
                elif input_mode == 999:
                     # This is rotation in place.
                    for _ in self.rotation(input_cmd): pass
                else: 
                    # This is pivot walking mode.
                    if input_mode != self.mode:
                        msg = f"Incompatibility at section: {section+1:02d}"
                        raise ValueError(msg)
                    # If no exception is occured, run the section.
                    for _ in self.feedforward_pivot(input_cmd, False): pass
        finally:
            # Reset the states to its initials.
            self.reset_state(*states[:-2])

########## test section ################################################
if __name__ == '__main__':
    specs = model.SwarmSpecs.robo3()
    xi = np.array([0,0,20,0,40,0])
    control = Controller(specs,xi,0,1)
    #control.reset_state(theta= np.deg2rad(-30))
    phi = np.deg2rad(135)
    sweep = np.deg2rad(30)
    mode = 2
    input_series = np.array([[10,0,1],
                             [0,0,1],
                             [12,0,-2],
                             [12*2,np.pi/2,0],
                             [10,0,2],
                             [12,0,0],
                             [0,np.pi/2,999],
                             [0,np.pi/4,1]])
    iterator = control.step_mode(mode, phi, last=False, line_up= True)
    iterator = control.step_tumble(phi, 2, last= False,line_up= True)
    iterator = control.step_pivot(phi, sweep, 10,last= False,line_up= True)
    iterator = control.pivot_walking_walk([2,0], last= False,line_up= True)
    iterator = control.mode_changing([0,phi,mode], last= False, line_up=True)
    iterator = control.tumbling([20,phi],last=False, line_up=True)
    iterator = control.feedforward_pivot([10,phi,1], last= False)
    control.compatibility_check(input_series)
    iterator = control.feedforward_line(input_series,has_last=True)
    for _ in iterator:
        i = control.get_state()
        #self.pos,self.theta,self.alpha,self.mode,self.posa,self.posb
        str_msg = (",".join(f"{elem:+07.2f}" for elem in i[0]) + "| "
                  +f"{np.rad2deg(i[1]):+07.2f},{np.rad2deg(i[2]):+07.2f}| "
                  +f"{i[3]:01d}"
                  )
        print(str_msg)
