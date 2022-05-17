#%%
########################################################################
# This files simulates consequetive transitions of milirobots. 
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from collections import deque
import numpy as np
import numpy.matlib

import model
import planner
########## Classes and functions #######################################
def consequtive(letters, specs, delay, steps, boundary):
    """
    This function plans for a given consequetive given positions
    of the milirobot.
    """
    UTOT = []
    U_delay = np.zeros((3,delay))
    # Get goal positions
    values = list(letters.values())
    # Plan position to position
    for i in range(len(values) - 1):
        # Set initial anf final position for current section.
        xi = values[i]
        xf = values[i+1]
        # Set up and solve the current step.
        planning = planner.Planner(specs, steps = steps,
                            solver_name='knitro', boundary=boundary)

        sol, U_raw, X_raw, P, UZ, U, X, isfeasible = planning.solve(xi, xf)
        # Append current section plan
        UTOT += [U]
        if delay >0:
            UTOT += [U_delay]
        # Update swarm state
    UTOT = np.hstack(UTOT)
    return UTOT
########## test section ################################################
if __name__ == '__main__':
    ## Specifications
    mode = 1
    angle = 0
    pivot_length = np.array([[10,9,8,7],[9,8,7,10],[8,7,10,9],[7,10,9,8]])
    letters = {}
    letters['start'] = np.array([0,0]+[20,0]+[40,0]+[60,0])
    letters['Y'] = np.array([0,0]+[0,20]+[-10,40]+[10,40]) 
    letters['F'] = np.array([0,0]+[0,50]+[25,50]+ [20,30])
    letters['D'] = np.array([0,0]+[0,40]+[30,35]+[30,5])
    letters['end'] = letters['start']
    transfer = np.array([0,0]*(len(letters['start'])//2))
    letters['start'] = letters['start'] + transfer
    xi = letters['start']
    xf = letters['end']
    ## Planning specs
    specs=model.SwarmSpecs(pivot_length, 10)
    steps = 2
    boundary = False
    last_section = True
    delay = 5
    U = consequtive(letters, specs, delay, steps, boundary)
    
    swarm = model.Swarm(xi, angle, mode, specs)
    # swarm.reset_state(xi,0,1)
    anim =swarm.simanimation(U,10000,boundary=boundary,
                                     last_section=last_section,save =False)
    #swarm.reset_state(xi,0,1)
    #swarm.simplot(U,500,boundary=boundary,last_section=last_section)
