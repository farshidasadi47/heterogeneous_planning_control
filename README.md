# heterogeneous_planning_control
This repository contains the code for the research paper
> Motion Planning for Multiple Heterogeneous Magnetic Robots Under Global Input

It performs motion planning for independent position control of multiple heterogeneous magnetic robots.

Videos related to experimental results to this paper can be seen below:

Example 1: Moving robots to a Y-shaped pattern with specified vertices | Example 3: Moving robots to S, M, U-shaped patterns letter by letter and back to initial positions | Example 4: Moving robots through a passage in three stages and aligning them on the other side
---|---|---
[![Video 1](https://img.youtube.com/vi/BO6eU7nUCqg/0.jpg)](https://www.youtube.com/watch?v=BO6eU7nUCqg) | [![Video 2](https://img.youtube.com/vi/xU0tjNfE78E/0.jpg)](https://www.youtube.com/watch?v=xU0tjNfE78E) | [![Video 3](https://img.youtube.com/vi/2hbBFitbgAU/0.jpg)](https://www.youtube.com/watch?v=2hbBFitbgAU)

## Structure of the code
The repository is a ROS2 package and the main code is in `swarm` folder.

The structure of this folder is:
```
swarm/
    helmholtz_arduino/
    paper/
    model.py
    planner.py
    leg_length.py
    localization.py
    closedloop.py
    rosclosed.py
```
- `helmholtz_arduino/` contain arduino code for the microcontroller that runs coil DC drivers. The code uses `micro-ROS` to communicate through serial port with computer. 
You may need to change it based on the microcontroller and DC driver that you use. To work with `micro-ros` see [micro-ROS for Arduino](https://github.com/micro-ROS/micro_ros_arduino).
- `paper/` contains some files process experimental data for the paper.
- `model.py` is a module that have a class that hold specifications of robotic groups and also provides a simulator to animate motion plans.
- `planner.py` is a module that contains a class that constructs motion planning problem according to the paper and solves it.
- `localization.py` is the image processing module that identifies and localizes robots.
- `closedloop.py` is the module that gets the motion plan and produces the low level motions needed to execute the command. 
It also contains methods to perform closedloop control as describged in the paper.
- `rosclosed.py` contains ROS2  processes to run the image processing and hardware communications. It uses all above modules in it.
The `localization.py`, `closedloop.py`, and `rosclosed.py` are modules that are only needed to run experiments involving hardware.

The `model.py` and `planner.py` modules can be used to perform simulations and will be described more.
## Dependencies
 The python dependencies of `model.py` and `planner.py` can be installed as:
 ```
 pip install numpy
 pip install matplotlib
 pip install casadi==3.5.5
 ```
 To run `planner.py` module, `Artelys Knitro version 10.3` software is also needed. To install this software please refer to [Artelys Knitro website](https://www.artelys.com/solvers/knitro/).
 
 ## Interfaces
 ### `model.py` module
 This module contains two classes:
 - `SwarmSpecs` which is mainly used to store specifictions of robot group.
 ```python
 import model  # Import model module.
# Defining specificiations of robot group.
pivot_length = np.array(
    [[10, 5, 5], [5, 5, 10]], dtype=float
)  # Rows are different modes.
tumble_length = 10.0
specs = model.SwarmSpecs(pivot_length, tumble_length)
 ```
 - `Swarm` which is mainly used to animate a given motion plan.
 ``` python
 # Create robot's object.
x_s = np.array([-20, 0, 0, 0, 20, 0], dtype=float)  # Initial positions.
mode_s = 1  # Initial mode of robots.
ang_s = 0  # initial angle of robots (Not important in simulation!!!).
swarm = model.Swarm(x_s, ang_s, mode_s, specs)  # Robotic group object.
# Motion plan: [displacement distance, displacement angle, displacement mode].T
input_series = np.array(
    [
        [70, np.pi / 2, 1],
        [70, -3 * np.pi / 4, 1],
        [10, -np.pi / 4, -2],
        [50, -np.pi / 2, 2],
        [50, np.pi / 4, 2],
        [50, np.pi / 4, 0],
        [10, np.pi / 4, -1],
    ],
    dtype=float,
).T

# Plot the motion plan.
plot_length = 10000  # How many steps to plot
boundary = True  # Draw boundaries of workspace.
last_section = False  # Draw all sections.
swarm.simplot(
    input_series, plot_length, boundary=boundary, last_section=last_section
)

# Animate motion plan.
last_section = True  # Only draw last section.
anim = swarm.simanimation(
    input_series, plot_length, boundary=boundary, last_section=last_section
)
 ```
### `planning.py` module
```python
import model  # Import model module.
import planner  # Import planning module.
# Defining specificiations of robot group.
pivot_length = np.array([[10, 5, 5], [5, 5, 10]], dtype=float)
tumble_length = 10.0
specs = model.SwarmSpecs(pivot_length, tumble_length)

# Setting workspace rectangular size.
ubx = 115  # Right side limit.
lbx = -115  # Left side limit.
uby = 90  # Upper side limit.
lby = -90  # Lower side limit.
rcoil = 100  # Radius of planar magnetic coil.
specs.set_space(ubx=ubx, lbx=lbx, uby=uby, lby=lby)

# Setting minimum distance.
specs.d_min = 10.0

# Defining boundary consitions
x_s = np.array([-30,  0,   0,  0, +30,  0],dtype=float)  # Initial position.
x_f = np.array([+30,+30,   0,  0,   0,+30],dtype=float)  # Final positions.
modes_s = 1  # Starting mode.
ang_s = 0  # Starting angle (Not important in simulation!!!).

# Defining planner object.
mode_sequence = [0, 1, 2]
N = 3  # Number of iterations of the mode sequence.
planning = planner.Planner(specs, mode_sequence, N, solver_name='knitro')

# Solve the planning problem.
_, _, _, _, _, _, U, _, isfeasible = planning.solve(x_s, x_f)

# Animate the solution.
swarm = model.Swarm(x_s, ang_s, mode_s, specs)
plot_length = 10000
boundary = True
last_section = True
anim = swarm.simanimation(
    U, plot_length, boundary=boundary, last_section=last_section
)
```
