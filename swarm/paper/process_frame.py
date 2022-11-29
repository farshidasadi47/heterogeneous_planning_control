#%%
########################################################################
# This files hold classes and functions that takes video frames of
# experimental tests and converts them to progression images for paper.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import sys

import cv2

import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mc
from matplotlib import rc
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
plt.rcParams['figure.figsize'] = [7.2,8.0]
plt.rcParams.update({'font.size': 11})
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times'
plt.rcParams['text.usetex'] = True
mpl.rcParams['hatch.linewidth'] = .5

np.set_printoptions(10000)

try:
    from swarm.model import define_colors
except ModuleNotFoundError:
    # Add parent directory and import modules.
    sys.path.append(os.path.abspath(".."))
    from model import define_colors

########## Functions ###################################################
class ProcessVideo:
    """
    This class gets video of swarm movement and converts it to snapshots
    of each step.
    """
    def __init__(self, path_list, length= 30, stride= None):
        """
        ----------
        Parameters
        ----------
        path_list: List denoting file paths
               path_list= [video_path, csv_path]
        """
        # From camera calibration, localization module. 
        self._p2mm= 0.48970705
        self._mm2p= 1/self._p2mm
        define_colors(self)
        self._colors = list(self._colors.keys())
        # Replace colors if you want.
        self._colors[self._colors.index("b")]= "deepskyblue"
        self.vid_path= path_list[0]
        self.csv_path= path_list[1]
        self.csv_data, self.sections= self._process_csv(self.csv_path)
        self.sections= self._combine_consequtives(self.sections)
        if  stride is None or stride == 0:
            self.dense_sections= self._populate_sections(self.csv_data,
                                                         self.sections,length)
        else:
            self.dense_sections= self._populate_sections_stride(self.csv_data,
                                                          self.sections,stride)
        self.frames= self._process_video(self.dense_sections, self.vid_path)
        self.blends= self._blend_frames(self.frames, self.dense_sections)
    
    def light(self, c, factor= 0.0):
        """Brightens colors."""
        if (c== 'k') and (factor != 0):
            # Since the transformation does not work on black.
            rgb= np.array([[[140,140,140]]],dtype= np.uint8)
        else:
            rgb= (np.array([[to_rgba(c)[:3]]])*255).astype(np.uint8)
        hls= cv2.cvtColor(rgb,cv2.COLOR_RGB2HLS_FULL)
        hls[0,0,1]= max(0, min(255,(1+factor)*hls[0,0,1]) )
        rgb= np.squeeze(cv2.cvtColor(hls,cv2.COLOR_HLS2RGB_FULL)
                        ).astype(float)/255
        return rgb.tolist()
        
    def _itemize(self, data_row):
        """
        Gets csv_data and returns a tuple of:
        input_cmd (3), mode, X, XI, XG, shape
        """
        # Headers
        """
        0  : counter,
        1-3: input_cmd,
        4  : theta,
        5  : alpha,
        6  : mode,
        7- : X, XI, XG, SHAPE
        """
        input_cmd= data_row[1:4]
        mode= data_row[6]
        X_SHAPE= np.reshape(data_row[7:], (4,-1))
        X=  X_SHAPE[0]
        XI= X_SHAPE[1]
        XG= X_SHAPE[2]
        SHAPE= X_SHAPE[3]
        return (input_cmd, mode, X, XI, XG, SHAPE)

    def _process_csv(self, csv_path, skip_mode_change= True):
        csv_data= np.loadtxt(csv_path, delimiter=',', skiprows=1)
        csv_data[:,0]= np.arange(len(csv_data)) # Reindex data.
        # Change points.
        diff_data= np.zeros((csv_data.shape[0],3),dtype=float)
        diff_data[0,0] = 1.0
        diff_data[1:,:]= np.diff(csv_data[:,[1,2,3]], axis= 0)
        change_idx= np.where(np.any(diff_data,axis=1))[0]
        change_idx= np.append(change_idx,len(csv_data)-1)
        sections= []
        for i in range(len(change_idx)-1):
            # Iterate in change_idx in pair.
            i_s= change_idx[i]
            i_e= change_idx[i+1] - 1
            mode_section= csv_data[i_s,3]
            # Exclude mode changes if requested.
            if (mode_section < 0) and skip_mode_change:
                continue
            section= []
            for ind in (i_s, i_e):
                data= csv_data[ind]
                data[0]= ind
                section.append(data)
            sections.append(section)
        return csv_data, sections

    def _combine_consequtives(self, sections):
        """
        Combine consequtive displacements of the same mode.
        """
        sections_n= []
        section= sections[0][:-1]
        for i in range(1,len(sections)):
            mode_p= sections[i-1][0][3]
            mode= sections[i][0][3]
            if mode_p == mode:
                # Consequtive sections with same mode.
                section.extend(sections[i][:-1])
            else:
                # Consequtive sections do not have same mode.
                section.append(sections[i-1][-1])
                sections_n.append(np.array(section,dtype=float))
                section= sections[i][:-1]
        section.append(sections[i][-1])
        sections_n.append(np.array(section,dtype=float))
        return sections_n                
    
    def _populate_sections_stride(self, csv_data, sections, stride= 3):
        dense_sections= []
        for section in sections:
            mode_section= section[0,3].astype(int)
            if mode_section == 999:
                # Staying still, no need to populating the points.
                dense_sections.append(section)
                continue
            # Populate by given stride.
            dense_section= []
            for i in range(len(section) - 1):
                # Iterate pair by pair.
                r_sect= section[i,1]
                i_s= section[i,0].astype(int)
                i_e= section[i+1,0].astype(int)
                alpha= csv_data[i_s:i_e+1,5]
                # Determine indexes where both pivot was unlifted.
                idxs0= np.diff(np.sign(abs(alpha)))
                idxs0= np.where(idxs0 >0)[0]+ i_s
                idxs= np.abs(np.diff(np.sign(alpha)))
                idxs= np.where(idxs>1)[0]+ i_s
                idxs= np.sort(np.concatenate((idxs,idxs0)))
                idxs[0]= i_s # To avoid showing inplace rotations.
                # Get necessary indexes.
                divs= len(idxs)/stride
                if (divs-int(divs)) > 0.4:
                    max_cnt= int(divs) + 1
                    indexes= np.arange(max_cnt)*stride
                    indexes= idxs[indexes]
                    indexes= np.append(indexes, (indexes[-1]+i_e)//2)
                else:
                    max_cnt= int(divs)
                    indexes= np.arange(max_cnt)*stride
                    indexes= idxs[indexes]
                # Get detailed data
                for ind in indexes[:-1]:
                    data= csv_data[ind]
                    data[0]= ind
                    dense_section.append(data)
            # Add end point.
            data= csv_data[i_e]
            data[0]= i_e
            dense_section.append(data)
            dense_sections.append(np.array(dense_section, dtype= float))
        return dense_sections

    def _populate_sections(self, csv_data, sections, length= 30):
        """
        Populates data between each sequence section.
        """
        dense_sections= []
        for section in sections:
            mode_section= section[0,3]
            if mode_section == 999:
                # Staying still, no need to populating the points.
                dense_sections.append(section)
                continue
            # Populate by given length.
            dense_section= []
            for i in range(len(section) - 1):
                # Iterate pair by pair.
                r_sect= section[i,1]
                i_s= section[i,0].astype(int)
                i_e= section[i+1,0].astype(int)
                # Calculate number of steps needed.
                n_step= np.ceil(r_sect/length).astype(int)
                step= np.ceil((i_e-i_s)/n_step).astype(int)
                for ind in range(i_s, i_e, step):
                    data= csv_data[ind]
                    data[0]= ind
                    dense_section.append(data)
            # Add end point.
            data= csv_data[i_e]
            data[0]= i_e
            dense_section.append(data)
            dense_sections.append(np.array(dense_section, dtype= float))
        return dense_sections
    
    def _cart2pixel(self,point):
        """Converts cartesian coordinte to pixel coordinate (x, y)."""
        pixel = np.zeros(2,dtype=int)
        pixel[0] = int(point[0]/self._p2mm) + self._center[0]
        pixel[1] = self._center[1] - int(point[1]/self._p2mm)
        return pixel

    def _find_center(self,frame):
        """
        Finds center of coordinate system based white border.
        """
        masked= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _,masked = cv2.threshold(masked,250,255,cv2.THRESH_BINARY)
        # Find the external contours in the masked image
        contours, hierarchy = cv2.findContours(masked, cv2.RETR_CCOMP,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        # Filter biggest external contour.
        external_areas = np.array([cv2.contourArea(contours[idx])
                             if elem >-1 else 0
                             for idx, elem in enumerate(hierarchy[0,:,2])])

        cnt= contours[np.argmax(external_areas)]
        x,y,w,h = cv2.boundingRect(cnt)
        xc= int(x+w/2)
        yc= int(y+h/2)
        # Frame parameters and center.
        f_height, f_width= frame.shape[:2]
        center= (xc, yc)
        return f_width, f_height, center

    def _process_video(self, sections, vid_path):
        """
        Gets frames for each section.
        """
        cap= cv2.VideoCapture(vid_path)
        frames= []
        for section in sections:
            frame_sec= []
            for line in section:
                idx= line[0].astype(int)
                # Set frame index and store frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, frame= cap.read()
                frame_sec.append(frame[:,:,::-1])
            frames.append(frame_sec)
        # Find center and limits
        frame= frame[:,:,::-1]
        self.f_width, self.f_height, self._center= self._find_center(frame)
        return frames

    def _blend_frames(self, frames, sections):
        """
        Creates blended snapshot of each section of movement.
        """
        h, w= self.f_height, self.f_width # Size of mask.
        radii= 17
        blends= []
        for frame_sec, section in zip(frames, sections):
            alpha= np.ones(len(frame_sec))*.4
            alpha[0]= 0.6
            alpha= np.linspace(0.3,0.9,len(frame_sec))
            alpha[-1]= 1.0
            master= frame_sec[-1].copy()
            for i, (frame, data) in enumerate(zip(frame_sec, section)):
                input_cmd, mode, X, XI, XG, SHAPE= self._itemize(data)
                al= alpha[i]
                # Make mask based on position of robots.
                mask= np.zeros((h,w), dtype= np.uint8)
                for pos in np.reshape(X,(-1,2)):
                    cent= self._cart2pixel(pos)
                    mask= cv2.circle(mask, cent, radii, 1, cv2.FILLED)
                weighted= cv2.addWeighted(frame, al, master,1-al,0)
                master[mask>0]= weighted[mask>0]
            blends.append(master)
        return blends

    def _simplot_set(self, ax, title, fontsize= None):
        """Sets the plot configuration."""
        ax.set_title(title, fontsize= fontsize, pad= 7)
        ax.set_xlabel('x axis',fontsize= fontsize)
        ax.set_ylabel('y axis',fontsize= fontsize)
        ax.axis("off")
        return ax

    def _plot_legends(self, ax, n_robot, light, path= True, shape= False):
        # Put legends.
        # Add robot legends.
        if path:
            ls= self._styles[1][1]
        else:
            ls= "none"
        legends= lambda ls,c,m,ms,mfc,mec,l: plt.plot([],[],
                                              linestyle=ls,
                                              linewidth= 2.5,
                                              color=c,
                                              marker=m,
                                              markersize= ms,
                                              markerfacecolor=mfc,
                                              markeredgecolor= mec,
                                              label= l)[0]
        # Robots and path.
        handles= [legends(ls, 
                          self.light(self._colors[robot]), 
                          self._markers[robot],
                          15,
                          self.light(self._colors[robot],light),
                          #self.light(self._colors[robot],light),
                          'k',
                          f"Robot {robot}"
                  ) for robot in range(n_robot)]
        labels= [f"Robot {robot}" for robot in range(n_robot)]
        handlelength= 3
        # Shape
        if shape:
            handlelength= 2
            ls= self._styles[2][1]
            handles+= [legends(ls, "orange", "o", 7, "orange",
                                   "orange", f"Target shape")]
            labels+= [f"Target shape"]
        ax.legend(handles= handles, labels= labels,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                fontsize= 24,
                framealpha= .8,
                facecolor= "w",
                handlelength= handlelength,
                handletextpad=0.05,
                labelspacing= 0.05,
        )
        return ax

    def _plot_markers(self,ax,pixels,section,light,length= 9999,legend=True):
        """
        Places markers on path based on given length.
        """
        for i in range(len(pixels)-1):
            sect= section[i]
            # Calculate number of steps needed.
            r_sect= sect[1]
            n_step= np.ceil(r_sect/length).astype(int)
            # Draw intermediary pixel points.
            pixes_s= pixels[i]
            pixes_e= pixels[i+1]
            pixes_step= ((pixes_e-pixes_s)/n_step).astype(int)
            for step in range(n_step):
                pixes= pixes_s + step*pixes_step
                for robot in range(int(len(pixes)/2)):
                    ax.plot(pixes[2*robot],pixes[2*robot+1], 
                        marker= self._markers[robot],
                        markerfacecolor= self.light(self._colors[robot],light),
                        markeredgecolor= "k",#self.light(self._colors[robot],.75),
                        markersize= 15,
                        zorder= 3,
                    )
        return ax

    def plot_single(self, blend, section, step, light, marker_spacing= 30,
                                                                   title=None):
        """
        Plots single section of movements.
        """
        fig, ax = plt.subplots(layout="constrained")
        # Plot the blended frame.
        ax.imshow(blend)
        # Calculate corresponding pixels..
        pixels= []
        for sect in section:
            input_cmd, mode, X, XI, XG, SHAPE= self._itemize(sect)
            pixes= []
            for point in np.reshape(X,(-1,2)):
                pixes.extend(self._cart2pixel(point))
            pixels.append(pixes)
        pixels= np.array(pixels, dtype= int)
        # Plot path, last part is arrow.
        for robot in range(int(len(pixes)/2)):
            ax.plot(pixels[:-1,2*robot], pixels[:-1,2*robot+1],
                color= self.light(self._colors[robot]),
                linewidth=2.5,
                linestyle= self._styles[1][1],
            )
        # Draw arrowed part.
        for robot in range(int(len(pixes)/2)):
            # Draw line.
            ax.annotate("",
                xy=((pixels[-1,2*robot]+.5)/self.f_width,
                    (self.f_height-pixels[-1,2*robot+1]-.5)/self.f_height),
                xycoords="axes fraction",
                xytext=((pixels[-2,2*robot]+.5)/self.f_width,
                        (self.f_height-pixels[-2,2*robot+1]-.5)/self.f_height),
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="arc3",
                    shrinkB=10,
                    color= self.light(self._colors[robot]),
                    linewidth= 2.5,
                    linestyle= self._styles[1][1],
                ),
            )
            # Draw the arrow head.
            ax.annotate("",
                xy=((pixels[-1,2*robot]+.5)/self.f_width,
                    (self.f_height-pixels[-1,2*robot+1]-.5)/self.f_height),
                xycoords="axes fraction",
                xytext=((pixels[-2,2*robot]+.5)/self.f_width,
                        (self.f_height-pixels[-2,2*robot+1]-.5)/self.f_height),
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>,head_length=1.0,head_width=0.4",
                    connectionstyle="arc3",
                    facecolor= self.light(self._colors[robot],light),
                    edgecolor=self.light("k",0),
                    linewidth= 0.0,
                ),
            )
        # Plot markers.
        ax= self._plot_markers(ax, pixels, section, light, )#marker_spacing)
        # Plot legends.
        ax= self._plot_legends(ax,int(len(pixes)/2), light)
        # Set plot border and title
        if title is None:
            mode= int(input_cmd[2])
            move_name= "pivot walking" if mode else "tumbling"
            title=f'Swarm transition step {step}: {move_name} in mode {mode:1d}'
        ax= self._simplot_set(ax, title,28)
        return fig, ax
    
    def _save_plot(self, fig, name):
        """Overwrites files if they already exist."""
        fig_name = os.path.join(os.getcwd(), f"{name}.pdf")
        fig.savefig(fig_name,bbox_inches='tight',pad_inches=0.05)
    
    def plot_transition(self, name= "example", title= None, 
                                               save= False, light= 0.6):
        FIG_AX= []
        sections= self.sections
        blends= self.blends
        for i, (blend, section) in enumerate(zip(blends, sections)):
            input_cmd, mode, X, XI, XG, SHAPE= self._itemize(section[-1])
            if np.any(input_cmd == 999):
                continue
            fig, ax= self.plot_single(blend, section, i, light, title=title)
            # Add frame to whole plot.
            fig.patch.set_edgecolor((0, 0, 0, 1.0))
            fig.patch.set_linewidth(2)
            if save:
                self._save_plot(fig, f"{name}_{i}")
            FIG_AX.append((fig,ax))
        return FIG_AX
    
    def plot_shape(self, name, title, light= 0.6,
                         save= False, desired= False, initial= False,shrink= 0):
        """
        Draws final pattern of the robots.
        """
        fig, ax = plt.subplots(layout="constrained")
        # Plot the blended frame.
        if initial:
            blend= self.frames[0][0]
            section= [self.sections[0][0]]
            desired= False
        else:
            blend= self.blends[-1]
            section= self.sections[-1]
        ax.imshow(blend)
        # Robots are stationary in last part.
        input_cmd, mode, X, XI, XG, SHAPE= self._itemize(section[-1])
        # Calculate pixels.
        pixes= []
        for point in np.reshape(X,(-1,2)):
            pixes.extend(self._cart2pixel(point))
        pixes= np.reshape(pixes, (-1,2)).astype(int)
        # Calculate shpae
        SHAPE= np.reshape(SHAPE[SHAPE<999],(-1,2)).astype(int)
        # Draw robot markers.
        for robot in range(int(len(pixes))):
            ax.plot(pixes[robot,0],pixes[robot,1], 
                marker= self._markers[robot],
                markerfacecolor= self.light(self._colors[robot],light),
                markeredgecolor= "k",#self.light(self._colors[robot],.75),
                markersize= 15,
                zorder= 3,
            )
        # Plot shapes.
        if desired:
            pixes= []
            for point in np.reshape(XG,(-1,2)):
                pixes.extend(self._cart2pixel(point))
            pixes= np.reshape(pixes, (-1,2)).astype(int)
        for inds in SHAPE:
            ax.plot(pixes[inds,0], pixes[inds,1],
                ls= self._styles[2][1], lw= 2.5, c="orange",
                marker= "o", markersize= 7,
                zorder= 1,
            )
            """ ax.annotate("",
                xy= ((pixes[inds[0],0]+.5)/self.f_width,
                        (self.f_height- pixes[inds[0],1]-.5)/self.f_height),
                xycoords="axes fraction",
                xytext=((pixes[inds[1],0]+.5)/self.f_width,
                        (self.f_height- pixes[inds[1],1]-.5)/self.f_height),
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="arc3",
                    shrinkA=shrink,
                    shrinkB=shrink, 
                    color= 'darkorange',
                    linewidth= 2.5,
                    linestyle= self._styles[2][1],
                ),
                zorder= 1,
            ) """
        # Plot legends.
        ax= self._plot_legends(ax,int(len(pixes)),light,
                               path=False,shape=desired)
        #
        if initial:
            title= f"Swarm transition: initial positions{title}"
        else:
            title= f"Swarm transition: desired positions{title}"
        ax= self._simplot_set(ax, title,28)
        # Add frame to whole plot.
        fig.patch.set_edgecolor((0, 0, 0, 1.0))
        fig.patch.set_linewidth(2)
        if save:
                self._save_plot(fig, f"{name}_desired")
        return [(fig, ax)]

def error_plot(name, title, xlabel, data, ticks, save= False):
    # Calculate
    data= abs(data)
    means= np.mean(data,axis= 1)
    std= np.std(data,axis=1)
    maxes= np.max(data,axis= 1)
    mins= np.min(data,axis= 1)
    xs= np.arange(len(means))
    # Plot
    fig, ax = plt.subplots(layout="constrained")
    ax.errorbar(xs, means, std, ls="", 
                ecolor="b", elinewidth=8, 
                marker= "D",mec= 'r', mfc="r", mew= 5.5,
                zorder= 2)
    ax.errorbar(xs, means,[means-mins,maxes-means], fmt=".k",
                capsize= 8, mew=1, elinewidth= 4,zorder= 1)
    # Set labels and title.
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks,fontsize= 32)
    ax.set_xlabel(xlabel,fontsize= 32)
    ax.set_ylabel(r"$e_{\mathrm{mm}}$" , fontsize=32)
    ax.yaxis.set_tick_params(labelsize=32)
    ax.set_title(title,fontsize= 38, pad= 11)
    # Set legends.
    handles, labels=[], []
    handles+= [plt.plot([],[],ls="",markerfacecolor= "r", marker="D", markersize=10)[0]]
    labels+=["Mean"]
    handles+= [plt.plot([],[],ls="-",color= "b", lw= 4)[0]]
    labels+=["One standard deviation range"]
    handles+= [plt.plot([],[],ls="-",color= "k", lw= 2)[0]]
    labels+=["Min\\Max range"]
    ax.legend(handles= handles, labels= labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize= 32,
            framealpha= .8,
            facecolor= "w",
            handletextpad=0.2,
            labelspacing= 0.05,
    )
    # Add frame to whole plot.
    fig.patch.set_edgecolor((0, 0, 0, 1.0))
    fig.patch.set_linewidth(2)
    # Save if requested
    if save:
        fig_name = os.path.join(os.getcwd(), f"{name}.pdf")
        fig.savefig(fig_name,bbox_inches='tight',pad_inches=0.1)
    return (fig,ax)

def progression():
    save= False
    light= 0.6
    # Tumbling
    file_dir= os.path.join(os.getcwd(), "fine_steps", "three", 
                                        "modes","tumble_longer")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    #
    process= ProcessVideo(path_lists, length= 28)
    name= "progression_0"
    title= "Swarm transition: Tumbling via mode 0"
    FIG_AX= process.plot_transition(name,title=title, save=save, light=light)
    # Mode 1
    file_dir= os.path.join(os.getcwd(), "fine_steps", "three", 
                                        "modes","mode_1")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    #
    process= ProcessVideo(path_lists, stride= 5)
    name= "progression_1"
    title= "Swarm transition: Pivot walking via mode 1"
    FIG_AX+= process.plot_transition(name,title=title, save=save, light=light)
    # Mode 2
    file_dir= os.path.join(os.getcwd(), "fine_steps", "three", 
                                        "modes","mode_2")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    #
    process= ProcessVideo(path_lists, stride= 5)
    name= "progression_2"
    title= "Swarm transition: Pivot walking via mode 2"
    FIG_AX+= process.plot_transition(name,title=title, save=save, light=light)
    plt.show()

def example1():
    save= False
    light= 0.6
    #
    file_dir= os.path.join(os.getcwd(), "fine_steps", "four", "QtoY2")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_1"
    FIG_AX= process.plot_transition(name,save= save,light= light)
    title= ""
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    plt.show()

def example2():
    save= False
    light= 0.6
    FIG_AX=[]
    # Initial repetition.
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "five", "QtoR_5iter", "QtoR4")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_2_i"
    title= ", first repetition"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # Fifth repetition.
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "five", "QtoR_5iter", "R2_3toR2_4")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_2_l"
    title= ", last repetition"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # Error plot
    data= np.array([
      [ -0.60,+12.42, +0.14, -1.68, +5.51,-14.78, +3.64, +6.19, -7.62, +0.44],
      [ -4.89, +2.90, -4.89, -5.51, -2.69, -5.11, -0.86, -1.64, +8.20, -2.85],
      [ +1.28, +2.18, -0.61, -1.47, -1.22, +3.71, +5.46, -2.62, -2.94, +2.41],
      [ -1.37, +5.26, -0.59, +0.13, +2.33, -1.80, -0.17, -1.88, -1.60, -1.87],
      [ +2.21, -1.26, -1.34, -4.92, -1.59, +1.14, -0.57, -1.25, -2.57, +3.06],
      #[ -0.92, +2.97, +1.61, -0.70, -1.34, -1.07, -1.68, -0.90, -0.52, +1.73],
    ])
    name="example_2_error"
    title="Errors versus repeating the algorithm"
    xlabel="Number of iterations $N$ in each repetition"
    ticks= np.array([4,2,2,2,2])
    FIG_AX+= error_plot(name, title, xlabel, data, ticks, save= save)
    plt.show()

def example3():
    save= False
    light= 0.6
    FIG_AX=[]
    # Q to S3S2
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "five", "SMU", "QtoS3S2")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_3_s"
    title= ", S shaped target"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # S to M4M2
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "five", "SMU", "StoM4M2")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_3_m"
    title= ", M shaped target"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # M to U4U3U2
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "five", "SMU", "MtoU4U3U2")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_3_u"
    title= ", U shaped target"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # U to Q3Q2
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "five", "SMU", "UtoQ3Q2")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_3_i"
    title= ", back to initial state"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    plt.show()

def example4():
    save= False
    light= 0.6
    FIG_AX=[]
    # Q to S3S2
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "four", "passage", "QtoR4")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_4_i"
    title= ""
    FIG_AX+= process.plot_shape(name, title, light, save, initial=True)
    # Q to R
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "four", "passage", "QtoR4")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_4_1"
    title= ", first stage"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # R to L
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "four", "passage", "R4toL1")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_4_2"
    title= ", second stage"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    # L to Q
    file_dir= os.path.join(os.getcwd(), "fine_steps",
                                        "four", "passage", "L1toQ4")
    path_lists= [os.path.join(file_dir,"logs.mp4"),
                 os.path.join(file_dir,"logs.csv")]
    process= ProcessVideo(path_lists, length= 30)
    name= "example_4_3"
    title= ", thrid stage"
    FIG_AX+= process.plot_shape(name, title, light, save, desired= True)
    plt.show()

########## test section ################################################
if __name__ == '__main__':
    example4()
