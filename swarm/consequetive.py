########################################################################
# This files simulates consequetive transitions of milirobots. 
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import numpy as np
import numpy.matlib

import model
import planner
########## Classes and functions #######################################
def consequtive(letters, pivot_separation, delay):
    """
    This function plans for a given consequetive given positions
    of the milirobot.
    """

    UTOT = []
    outer = 4
    boundary = True
    U_delay = np.zeros((3,delay))
    # Get goal positions
    values = list(letters.values())
    # Set up the swarm
    swarm_specs=model.SwarmSpecs(pivot_separation, 5, 10)
    swarm = model.Swarm(values[0], 0, 1, swarm_specs)
    # Plan position to position
    for i in range(len(values) - 1):
        # Set initial anf final position for current section.
        xf = values[i+1]
        # Set up and solve the current step.
        planning = planner.Planner(swarm, n_outer = outer)
        planning.get_optimization(solver_name='knitro', boundary=False)
        sol, U_sol, X_sol, P_sol, UZ, U, X = planning.solve_optimization(xf,boundary)
        # Append current section plan
        UTOT += [U]
        if delay >0:
            UTOT += [U_delay]
        # Update swarm state
        swarm.simulate(U)
    UTOT = np.hstack(UTOT)
    return UTOT
########## test section ################################################
if __name__ == '__main__':
    ## Specifications
    mode = 1
    angle = 0
    pivot_separation = np.array([[10,9,8,7],[9,8,7,10],[8,7,9,10]])
    letters = {}
    letters['start'] = np.array([0,0]+[20,0]+[40,0]+[60,0])
    letters['Y'] = np.array([0,0]+[0,20]+[-10,40]+[10,40]) 
    letters['F'] = np.array([0,0]+[0,50]+[25,50]+ [20,30])
    letters['D'] = np.array([0,0]+[0,40]+[30,35]+[30,5])
    letters['end'] = letters['start']
    transfer = np.array([0,0]*(len(letters['start'])//2))
    xi = letters['start'] + transfer
    xf = letters['end']
    ## Planning specs
    outer = 4
    boundary = True
    last_section = True
    delay = 5
    U = consequtive(letters, pivot_separation, delay)
    
    swarm_specs=model.SwarmSpecs(pivot_separation, 5, 10)
    swarm = model.Swarm(xi, angle, mode, swarm_specs)
    
    # swarm.reset_state(xi,0,1)
    anim =swarm.simanimation(U,10000,boundary=boundary,last_section=last_section,save =False)
    
    #swarm.reset_state(xi,0,1)
    #swarm.simplot(U,500,boundary=boundary,last_section=last_section)



