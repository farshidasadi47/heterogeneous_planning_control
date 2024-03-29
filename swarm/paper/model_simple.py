#%%
########################################################################
# This files hold classes and functions that simulates the milirobot 
# system without considering mode change.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import sys
from itertools import combinations
from collections import deque
from math import remainder

import numpy as np
np.set_printoptions(precision=6, suppress=False)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
plt.rcParams['figure.figsize'] = [7.2,8.0]
plt.rcParams.update({'font.size': 11})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False
mpl.rcParams['hatch.linewidth'] = .5

try:
    from swarm.model import SwarmSpecs
    from swarm.planner import Planner
except ModuleNotFoundError:
    # Add parent directory and import modules.
    sys.path.append(os.path.abspath(".."))
    from model import SwarmSpecs
    from planner import Planner
########## Functions ###################################################
class Swarm:
    """This class holds current state of swarm of milirobots."""
    
    def __init__(self, position, specs: SwarmSpecs):
        self.specs = specs
        self.reset_state(position)
        self._simulate_result = None

    def reset_state(self, position):
        position= np.array(position, dtype= float)
        msg = "Position does not match number of the robots in spec."
        assert (position.shape[0]//2 == self.specs.n_robot), msg
        self.position = position
    
    def update_state(self, u: np.ndarray):
        """
        This function updates the position, angle, and mode of swarm
        of milirobots based on given input.
        ----------
        Parameters
        ----------
        u: numpy.ndarray
           A numpy array as [r,phi]
           r:     Distance to travel
           phi: Angle to travel
           mode: int
              The mode motion is performed.
        """
        n_mode = self.specs.n_mode
        u= np.array(u)
        r = u[0]
        phi = u[1]
        mode = int(u[2])
        # Modify input based on requested motion mode.
        if mode > 0:
            # Pivot walking
            B = self.specs.B[mode,:,:]
        else:
            B = self.specs.B[0,:,:]
        # Convert input to cartesian.
        u[0] = r*np.cos(phi)
        u[1] = r*np.sin(phi)
        # Update states of swarm of milirobots
        self.position = self.position + np.dot(B,u[:-1])

    def simulate(self, input_series, position = None, step_size= None):
        """Simulates the swarm for a given logical series of input."""
        msg = 'Input series should be a 2D numpy array.'
        assert (input_series.ndim == 2), msg
        if position is None:
            position = self.position
        #
        self.reset_state(position)
        cum_position = []
        cum_input = []
        # Execute input section by secton.
        for section in input_series:
            cmd_r= section[0]
            cmd_phi= section[1]
            cmd_mode= int(section[2])
            if cmd_mode>=0:
                # Divide the current section to smaller pieces.
                if step_size is None:
                    steps= np.array([cmd_r])
                else:
                    steps= np.arange(0,cmd_r,step_size)
                    steps= np.append(steps, cmd_r)
                    steps= np.diff(steps)
                    if steps.size == 0: # This lets simulate zero inputs.
                        steps = np.array([0.0])
                # Implementing current input section in smaller steps
                sec_position= [self.position]
                sec_input= []
                for step in steps:
                    step_input = np.array([step,cmd_phi, cmd_mode])
                    # Applying current step
                    self.update_state(step_input)
                    sec_position.append(self.position)
                    sec_input.append(step_input)
                # Add zero step for convinience.
                step_input= step_input.copy()
                step_input[0]= 0.0
                sec_input.append(step_input)
                # Store the section's results.
                cum_position.append(np.array(sec_position,dtype= float))
                cum_input.append(np.array(sec_input,dtype= float))
        self._simulate_result = (cum_position, cum_input)
        return cum_position, cum_input

    def _regroup_sim_result(self, paired= False, n_section= 1):
        """
        This function joins consequtive motion sequence based on having
        same mode consequtively, or a given number of sequences.
        """
        # Get the simulation results.
        cum_position, cum_input = self._simulate_result
        g_position= []
        g_input= []
        # Combine consequtive sections with similar mode if requested.
        if paired:
            # Find start and end indexes.
            mode_seq= [int(i[0,2]) for i in cum_input]
            distinct_mode_idx= []
            i, id_s= 0, 0 
            for i in range(1,len(mode_seq)):
                if mode_seq[i]-mode_seq[i-1]:
                    # If mode has changed.
                    distinct_mode_idx.append((id_s,i-1))
                    id_s= i
            distinct_mode_idx.append((id_s,i))
            # Combine sections.
            for idx_s, idx_e in distinct_mode_idx:
                sec_position= cum_position[idx_s]
                sec_input= cum_input[idx_s]
                for sec_pos, sec_inp in zip(cum_position[idx_s+1:idx_e+1],
                                           cum_input[idx_s+1:idx_e+1]):
                    sec_position= np.vstack((sec_position[0:-1,:], sec_pos))
                    sec_input= np.vstack((sec_input[0:-1,:], sec_inp))
                g_position.append(sec_position)
                g_input.append(sec_input)
            cum_position= g_position
            cum_input= g_input
            g_position= []
            g_input= []
        # Combine each N sections in requested.
        indexes= list(range(0,len(cum_input), n_section)) + [len(cum_input)]
        indexes= [(i,j) for i,j in zip(indexes[:-1],indexes[1:])]
        # Combine sections.
        for idx_s, idx_e in indexes:
            sec_position= cum_position[idx_s]
            sec_input= cum_input[idx_s]
            for sec_pos, sec_inp in zip(cum_position[idx_s+1:idx_e],
                                        cum_input[idx_s+1:idx_e]):
                sec_position= np.vstack((sec_position[0:-1,:], sec_pos))
                sec_input= np.vstack((sec_input[0:-1,:], sec_inp))
            g_position.append(sec_position)
            g_input.append(sec_input)
        return g_position, g_input

    def _simplot_set(self, ax, fontsize= None, title= None,
                           showx=True, showy=True):
        """Sets the plot configuration."""
        title= 'Swarm transition' if title is None else title
        xlabel= 'x axis' if showx else None
        ylabel= 'y axis' if showy else None
        ax.set_title(title, fontsize= fontsize, pad= 7)
        ax.set_xlabel(xlabel, fontsize= fontsize)
        ax.set_ylabel(ylabel, fontsize= fontsize)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(grid_alpha=0.5,labelsize= fontsize, 
                       labelbottom=showx, labelleft=showy)
        ax.grid()

    def _plot_arrow(self, ax, from_pos, to_pos, color):
        ax.annotate("",
            xy=to_pos,
            xytext=from_pos,
            arrowprops=dict(
                arrowstyle="-|>,head_length=1.5,head_width=0.5",
                connectionstyle="arc3",
                color= color,
                linewidth= 0.0,
            ),
        )
        return ax
    
    def _plot_circle(self, ax, xy, color, lw=2.0, zorder=0):
        circ = plt.Circle(xy,
                          radius=self.specs.tumbling_length/2,
                          fill=False,
                          ec=to_rgba(color,0.6),
                          ls = (0, (5, 5)),
                          lw=lw,
                          zorder=zorder
               )
        ax.add_patch(circ)
        return ax
    
    def _plot_colliding(self, ax, section, draw_circle):
        res_max = 1
        tumbling_length = self.specs.tumbling_length
        # Search for colliding parts of path.
        n_points = int(np.linalg.norm(section[0,:2]-section[-1,:2])/res_max)
        points = np.linspace(section[0], section[-1], n_points + 1)
        points = np.transpose(points.T.reshape(self.specs.n_robot, 2, -1),
                              (0,2,1))
        robots_colliding= set()
        for pairs in self.specs.robot_pairs:
            distances = np.linalg.norm(points[pairs[0]]-points[pairs[1]], axis=1)
            ind_colliding = np.argwhere(distances < tumbling_length).flatten()
            if ind_colliding.size:
                for robot in pairs:
                    robots_colliding.add(robot)
                    if draw_circle:
                        """ # Start of collision.
                        self._plot_circle(ax,
                                            points[robot][ind_colliding[0]],
                                            self.specs._colors[robot]
                                            ) """
                    # End of collision.
                    """ self._plot_circle(ax,
                                      points[robot][ind_colliding[-1]],
                                      self.specs._colors[robot]
                                      ) """
                    # Collision section.
                    xy = points[robot][ind_colliding[[0,-1]]]
                    c=to_rgba(self.specs._colors[robot], 0.4)
                    ax.plot(xy[:,0], 
                            xy[:,1],
                            c=c,
                            linewidth=15,
                            solid_capstyle="butt",
                            marker="o",mfc=c,ms=4,
                            zorder=0,
                            )
        return robots_colliding

    def single_plot(self, data_position, data_input,
                          legend= True, showx=True, showy=True,
                          lbx= None, ubx= None, lby=None, uby= None,
                          title= None, save= False):
        """Plots a 2D series of given position."""
        title= 'Swarm transition' if title is None else title
        tumbling_length = self.specs.tumbling_length
        alpha_s= 0.1
        alpha_f= 0.7
        modes_used= set()
        legend_marker= lambda m,c,l: plt.plot([],[],marker=m, color=c,
                                                    markerfacecolor=c,ms=6,
                                                    ls="none", label= l)[0]
        legend_line= lambda s,c,l: plt.plot([],[],ls=s,color=c,label=l,lw=2)[0]
        # Geometry of robot symbols.
        aspect_ratio = 2
        h = np.sqrt(tumbling_length**2/(aspect_ratio**2+1))
        w = aspect_ratio*h
        legend_robot= lambda c,a,ha: plt.Rectangle([0,0] ,
                                 width= w, height= h,
                                 linestyle='-', linewidth=1.5,
                                 edgecolor= to_rgba(c,1.0),
                                 facecolor = to_rgba(c, a),
                                 hatch= ha,
                                 )
        # Legend for collisions.
        legend_collision= lambda c: plt.plot([],[],color=c,lw=12,alpha=0.4,
                                             solid_capstyle="butt")[0]
        # Set the figure properties
        fig, ax = plt.subplots(constrained_layout=True)
        fontsize= 24
        self._simplot_set(ax, fontsize, title, showx, showy)
        # Draw lines
        for robot in range(self.specs.n_robot):
            section_iterator= zip(data_position, data_input)
            sec_position, sec_input= next(section_iterator)
            ang= sec_input[0,1]- np.pi/2
            mode= int(sec_input[0,2])
            modes_used.add(mode)
            # Draw robot at start.
            x = -(w*np.cos(ang) - h*np.sin(ang))/2
            y = -(w*np.sin(ang) + h*np.cos(ang))/2
            rect = plt.Rectangle([sec_position[0,2*robot] + x,
                                  sec_position[0,2*robot+1] + y] ,
                        width= w, height= h, angle= np.rad2deg(ang),
                        linestyle='-', linewidth=1.5,
                        edgecolor= to_rgba(self.specs._colors[robot],1.0),
                        facecolor = to_rgba(self.specs._colors[robot],alpha_s),
                        zorder=2,
                        )
            ax.add_patch(rect)
            # Draw first section path.
            ax.plot(sec_position[:,2*robot], sec_position[:,2*robot+1],
                    color= self.specs._colors[robot],
                    linewidth=2,
                    linestyle= self.specs._styles[robot][1],
                    marker= self.specs._markers[mode],
                    markerfacecolor= self.specs._colors[robot],ms= 8)
            """ ax = self._plot_arrow(ax,
                                  sec_position[-2, 2*robot:2*robot+2],
                                  sec_position[-1, 2*robot:2*robot+2],
                                  self.specs._colors[robot]) """
            # Draw other sections.
            for sec_position, sec_input in section_iterator:
                ang= sec_input[-1,1]- np.pi/2
                mode= int(sec_input[0,2])
                modes_used.add(mode)
                ax.plot(sec_position[:,2*robot], sec_position[:,2*robot+1],
                        color = self.specs._colors[robot],
                        linewidth=2,
                        linestyle= self.specs._styles[robot][1],
                        marker= self.specs._markers[mode],
                        markerfacecolor= self.specs._colors[robot])#'none')
            ax = self._plot_arrow(ax,
                                    sec_position[-2, 2*robot:2*robot+2],
                                    sec_position[-1, 2*robot:2*robot+2],
                                    self.specs._colors[robot])
            # Draw robot at the end.
            ang= sec_input[-1,1]- np.pi/2
            x = -(w*np.cos(ang) - h*np.sin(ang))/2
            y = -(w*np.sin(ang) + h*np.cos(ang))/2
            rect = plt.Rectangle([sec_position[-1,2*robot] + x,
                                  sec_position[-1,2*robot+1] + y] ,
                        width= w, height= h, angle= np.rad2deg(ang),
                        linestyle='-', linewidth=1.5,
                        edgecolor= to_rgba(self.specs._colors[robot],1.0),
                        facecolor = to_rgba(self.specs._colors[robot],alpha_f),
                        #hatch= 'xx',
                        zorder=2,
                                 )
            ax.add_patch(rect)
        # Draw colliding sections.
        draw_circle=True
        robots_colliding = set()
        for sec_position in data_position:
            robots_colliding |= self._plot_colliding(ax, sec_position,
                                                     draw_circle)
            draw_circle &= len(robots_colliding)
        # Calculate plot boundaries
        lbxx, ubxx, lbyy, ubyy= np.inf, -np.inf, np.inf, -np.inf
        for sec_position in data_position:
            # Update space limits
            lbxx= min(lbxx,sec_position[:,::2].min()-(tumbling_length/2+1))
            ubxx= max(ubxx,sec_position[:,::2].max()+(tumbling_length/2+1))
            lbyy= min(lbyy,sec_position[:,1::2].min()-(tumbling_length/2+1))
            ubyy= max(ubyy,sec_position[:,1::2].max()+(tumbling_length/2+1))
        lbx= lbxx if lbx is None else lbx
        ubx= ubxx if ubx is None else ubx
        lby= lbyy if lby is None else lby
        uby= ubyy if uby is None else uby
        ax.set_xlim(lbx,ubx)
        ax.set_ylim(lby,uby)
        # Add robot legends.
        handles= [legend_line(self.specs._styles[robot][1],
                   self.specs._colors[robot], 
                   f"Robot {robot}") for robot in range(self.specs.n_robot)]
        labels= [f"Robot {robot}" for robot in range(self.specs.n_robot)]
        # Add mode legends.
        handles+= [tuple(legend_marker(self.specs._markers[mode],
                        self.specs._colors[robot], 
                        f"Mode {mode}") for robot in range(self.specs.n_robot))
                   for mode in modes_used
                   ]
        labels+= [f"Mode {mode}" for mode in modes_used]
        # Add start legends.
        handles+= [tuple(legend_robot(self.specs._colors[robot],alpha_s,None)
                         for robot in range(self.specs.n_robot))]
        labels+= ["Start"]
        # Add end legends.
        handles+= [tuple(legend_robot(self.specs._colors[robot],alpha_f,None)#'xxxx')
                         for robot in range(self.specs.n_robot))]
        labels+= ["End"]
        # Add collision related legends.
        if len(robots_colliding):
            handles+= [tuple(legend_collision(self.specs._colors[robot])
                            for robot in robots_colliding)]
            labels+= ["Colliding"]
        if legend:
            ax.legend(handles= handles, labels= labels,
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       fontsize= fontsize,
                       )
        # Saving figure if requested.
        if save:
            # Set file name for saving animation.
            index_for_saving = 1
            fig_name = title+f"_{index_for_saving:02d}.pdf"
            # Check if the current file name exists in the directory.
            while os.path.isfile(fig_name):
                # Increase file number index until no file with such
                # name exists.
                index_for_saving += 1
                fig_name = title+f"_{index_for_saving:02d}.pdf"
            fig.savefig(fig_name,bbox_inches='tight', pad_inches= 0.05)
        return fig, ax

def main():
    specs = SwarmSpecs(np.array([[10,5,5],[5,5,10]]), 10)
    xi = np.array([-20,0,0,0,20,0])
    mode = 1
    swarm = Swarm(xi, specs)
    input_series = np.array([[60,np.pi/2,1],
                             [70,np.pi,1],
                             [70,-np.pi/2,0],
                             [70,np.pi,2],
                             [70,np.pi/2,2],
                             ])
    step_size= 20
    cum_position, cum_input= swarm.simulate(input_series, xi, step_size)
    g_position, g_input= swarm._regroup_sim_result(paired=True, n_section=1)
    title= None
    fig, ax= swarm.single_plot(cum_position,cum_input,legend= True,title= title,save= False)
    plt.show()

def case2():
    specs = SwarmSpecs(np.array([[10,5,5],[5,5,10]]), 10)
    save= False
    specs.d_min= 14
    mode_sequence= [1,1,2,2]*1
    xi = np.array([000,000, 000,-20, 000,+20],dtype=float)
    xf = np.array([-53,+52, -40,+ 6, -67,+46],dtype=float)
    swarm = Swarm(xi, specs)
    step_size= 10
    planner = Planner(specs, mode_sequence , steps = 1,
                      solver_name='knitro', feastol= 0.1,boundary= True)
    #
    UU_raw= np.array([[58.137767, 2.034444, 1],
                      [27       , 3.141593, 2],
                     ],dtype= float)
    _, UU_raw, _, _ , _, _= planner.solve_unconstrained(xi,xf)
    UU_raw= UU_raw.T
    UU_raw= UU_raw[UU_raw[:,2]>0]
    print(UU_raw)
    cum_position, cum_input= swarm.simulate(UU_raw, xi, step_size)
    title= "(a): Controllability Solution."
    fig1, ax1= swarm.single_plot(cum_position,cum_input,
                                 legend= True,
                                 lby= -27, uby= 60, lbx= -77,
                                 title= title,save= save)
    print(swarm.position)
    #
    U_raw= np.array([[43.123117,  2.371032, 1],
                     [22.512195,  1.349482, 1],
                     [18.75    , -2.214297, 2],
                     [21.75    ,  2.38058 , 2],
                    ],dtype= float)
    _, _, U_raw, X_raw, _, _, _, _, _ = planner.solve(xi, xf)
    U_raw= U_raw.T
    swarm.reset_state(xi)
    cum_position, cum_input= swarm.simulate(U_raw, xi, step_size)
    title= "(b): Divided, Part 1."
    fig2,ax2= swarm.single_plot(cum_position[0:2],cum_input[0:2],legend= True,
                                lby= -27, uby= 60, lbx= -50, showy= False,
                                title= title,save= save)
    title= "(c): Divided, Part 2."
    fig2,ax2= swarm.single_plot(cum_position[2:4],cum_input[2:4],legend= True,
                                lby= -27, uby= 60, lbx= -77,showy= False,
                                title= title,save= save)
    print(swarm.position)
    plt.show()

def case3():
    specs = SwarmSpecs(np.array([[10,5,5],[5,5,10]]), 10)
    save= False
    specs.d_min= 14
    mode_sequence= [1,2,1,2]*1
    xi = np.array([ 000,000,000,-30, 000,+30],dtype=float)
    xf = np.array([+19,+34, -23,+29, - 8,+53],dtype=float)
    xf = np.array([+46,+41, -17,+35, +23,+64],dtype=float)
    xf = np.array([-17,+30, -11,-29, -16,+ 3],dtype=float)
    swarm = Swarm(xi, specs)
    step_size= 10
    planner = Planner(specs, mode_sequence , steps = 1,
                      solver_name='knitro', feastol= .01, boundary= True)
    #
    UU_raw= np.array([[59.228372,  1.774814, 1],
                      [28.442925, -1.747505, 2],
                      ],dtype= float)
    _, UU_raw, _, _ , _, _= planner.solve_unconstrained(xi,xf)
    UU_raw= UU_raw.T
    UU_raw= UU_raw[UU_raw[:,2]>0]
    print(UU_raw)
    cum_position, cum_input= swarm.simulate(UU_raw, xi, step_size)
    title= "(a): Controllability Solution."
    fig1, ax1= swarm.single_plot(cum_position,cum_input,legend= True,
                                 lby= -36, uby= 63, lbx= -61,
                                 title= title,save= save)
    print(swarm.position)
    xf= np.round(swarm.position)
    #
    U_raw= np.array([[18.954072,  1.085278, 1],
                     [14.012177, -0.478136, 2],
                     [46.205665,  2.038839, 1],
                     [27.725384, -2.251132, 2],
                     ],dtype= float)
    _, _, U_raw, X_raw, _, _, _, _, _ = planner.solve(xi, xf)
    U_raw= U_raw.T
    swarm.reset_state(xi)
    step_size= 10
    cum_position, cum_input= swarm.simulate(U_raw, xi, step_size)
    title= "(b): Divided and Rearranged, Part 1."
    fig2,ax2= swarm.single_plot(cum_position[0:2],cum_input[0:2],legend= True,
                                lby= -36, uby= 63, lbx= -41, showy= False,
                                title= title,save= save)
    title= "(c): Divided and Rearranged, Part 2."
    fig2,ax2= swarm.single_plot(cum_position[2:4],cum_input[2:4],legend= True,
                                lby= -36, uby= 63, lbx= -58, showy= False,
                                title= title,save= save)
    print(swarm.position)
    plt.show()

########## test section ################################################
if __name__ == '__main__':
    case2()
