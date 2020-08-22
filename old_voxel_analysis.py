import sys
### Modify sys path if the tof project folder is not in PATH 
sys.path.append("D:\\tof")
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Frame_data():
    def __init__(self,all_data,step=30,axis_name=" ",lower_thresh=20):
        self.all_data=all_data
        data_start=all_data[0]
        
        self.hist_min=data_start.min()
        self.hist_max=data_start.max()
        self.step=step
        self.lower_thresh=lower_thresh

        mask=data_start.flatten()>self.lower_thresh
        self.data=data_start.flatten()[mask]
        
        self.bins=np.arange(self.hist_min,self.hist_max+2*self.step,self.step)
        self.count, self.bins=np.histogram(self.data,bins=self.bins,density=False)

    def update_data(self,frame_num,mode="continous"):
        data_update=self.all_data[frame_num]
        mask=data_update.flatten()>self.lower_thresh
        data_update=data_update.flatten()[mask]
        check1=self.hist_min>data_update.min(); check2=self.hist_max<data_update.max()
        
        if check1:
            self.hist_min=data_update.min()
        if check2:
            self.hist_max=data_update.max()
        if check1 or check2:
            self.bins=np.arange(self.hist_min,self.hist_max,self.step)
        if mode=="continous":
            self.data=np.hstack([self.data,data_update])
        elif mode=="replace":
            self.data=data_update
        self.count, self.bins=np.histogram(self.data,bins=self.bins,density=False)

    @staticmethod    
    def animate(i):
        print(i)
        self.update_data()
        plt.xticks(self.bins[:-1], rotation=90)
        plt.hist(self.data, bins=bins[:-1], density=False, facecolor='g', alpha=0.9)
        plt.xlim(self.bins.min(), self.bins.max())
        plt.ylim(self.data.min(), self.data.max(),self.step)
        #sns.boxplot(self.data, ax=self.bins)
        #sns.distplot(self.data, ax=self.count, bins=self.bins) 

class Graph_Anim(Frame_data):
    def __init__(self,all_data,step=30,axis_name=" ",lower_thresh=20):
        self.fdata=Frame_data(all_data,step=step,axis_name=axis_name,lower_thresh=lower_thresh)

    def start_graph(self,figsize=(9,8)):
        
        self.fig, ax= plt.subplots( figsize=figsize)
        plt.xlabel("Amplitude Range")
        plt.ylabel('Count')
        plt.title('Histogram Test')
        plt.xticks(self.fdata.bins, rotation=90)
        plt.xlim(self.fdata.bins.min(), self.fdata.bins.max())
        plt.ylim(self.fdata.count.min(), self.fdata.count.max())
        plt.grid(True)

    def update_graph(self):
        frames=self.fdata.all_data.shape[0]
        ims=[]
        for frame in range(frames):
            if frame!=0:
                #print(frame,len(ims))
                self.fdata.update_data(frame,mode="continous")
                plt.title('Histogram Test'+str(frame))
                plt.xticks(self.fdata.bins, rotation=90)
                plt.xlim(self.fdata.bins.min(), self.fdata.bins.max())
                plt.ylim(self.fdata.count.min(), self.fdata.count.max())

                #print(self.fdata.data.shape,"data")
                #print(self.fdata.bins.shape,"bins")
                _,_,objt=plt.hist(self.fdata.data,
                                bins=self.fdata.bins, 
                                density=False, 
                                facecolor='g', 
                                alpha=0.9)
                #from IPython import embed; embed()
                pack=(objt)
                #from IPython import embed; embed()
                ims.append(pack)

        im_ani = animation.ArtistAnimation(self.fig, ims, 
                                           interval=33,
                                           repeat_delay=500,
                                           blit=True)
        plt.show()

class Dual_Param_Analysis():
    def __init__(self,A_data,B_data,A_name,B_name,stepA,stepB):
        self.hist_minA=A_data.min()
        self.hist_maxA=A_data.max()
        self.hist_binsA=np.arange(self.hist_minA,self.hist_maxA,stepA)
        self.A_data=A_data
        self.A_name=A_name
        self.B_name=B_name
        self.bin_pair=[]
        for i in range(len(self.hist_binsA)):
            if i+1<len(self.hist_binsA):
                self.bin_pair.append([self.hist_binsA[i],self.hist_binsA[i+1]])
        self.B_data=B_data

    def _bins_mask(self,bin_min,bin_max):
        lower_mask=self.A_data>=bin_min
        high_mask=self.A_data<=bin_max
        final_mask=np.logical_and(lower_mask,high_mask)
        return final_mask
    
    def _mask_count(self,final_mask):
        mask_data=self.B_data[final_mask]
        if len(mask_data.shape)==1:
            mask_count=len(mask_data)
        elif len(mask_data.shape)==2:
            mask_count=mask_data.shape[0]*mask_data.shape[1]
        elif len(mask_data.shape)==3:
            mask_count=mask_data.shape[0]*mask_data.shape[1]*mask_data.shape[2]
        return mask_count

    def _apply_mask(self,final_mask):
        mask_data=self.B_data[final_mask]
        mask_mean=mask_data.mean()
        mask_std=mask_data.std()
        return mask_mean,mask_std
    
    def loop_mask(self):
        final_count=[]
        for pair in self.bin_pair:
            final_mask=self._bins_mask(pair[0],pair[1])
            mask_count=self._mask_count(final_mask)
            if mask_count!=0:
                mask_mean,mask_std=self._apply_mask(final_mask)
                final_count.append([pair[1],mask_count,mask_mean,mask_std])
        return final_count
    
    def plot(self,final_count):
        fct=np.array(final_count)
        x=fct[:,0]
        y=fct[:,2]
        yerr=fct[:,3]
        _, ax = plt.subplots(figsize=(7, 4))
        ls = 'dotted'
        title_name=(self.A_name+" vs "+self.B_name+" Mean+-Std")
        ax.set_title(title_name)
        for i in range(len(fct)):
            ax.errorbar(x,y,yerr=yerr,marker='o',linestyle=ls)
                            
        plt.show()

class Triple_Param_Analysis():
    
    def __init__(self,A_pack,B_pack,C_pack):
        A_data,A_name,stepA=A_pack
        B_data,B_name,stepB=B_pack
        C_data,C_name,stepC=C_pack
        ##### A pack
        self.hist_minA=A_data.min()
        self.hist_maxA=A_data.max()
        self.hist_binsA=np.arange(self.hist_minA,self.hist_maxA,stepA)
        ##### B pack
        self.hist_minB=B_data.min()
        self.hist_maxB=B_data.max()
        self.hist_binsB=np.arange(self.hist_minB,self.hist_maxB,stepB)
        
        self.heat_map_count=np.zeros([len(self.hist_binsA),len(self.hist_binsB)])
        self.heat_map_mean=np.zeros([len(self.hist_binsA),len(self.hist_binsB)])
        self.heat_map_std=np.zeros([len(self.hist_binsA),len(self.hist_binsB)])
        #from IPython import embed; embed()
        self.final_count_3d=[]
        self.A_data=A_data
        self.B_data=B_data
        self.C_data=C_data
        self.A_name=A_name
        self.B_name=B_name
        self.C_name=C_name
    
    def param_calc(self):
        
        for num1, bin1 in enumerate(self.hist_binsA):
            if num1+1<len(self.hist_binsA):
                for num2, bin2 in enumerate(self.hist_binsB):
                    if num2+1<len(self.hist_binsB):
                        bin1_n=self.hist_binsA[num1+1]
                        bin2_n=self.hist_binsB[num2+1]
                        coord=[num1,num2]

                        ###### _bin_mask
                        lower_maskA=self.A_data>=bin1; high_maskA=self.A_data<=bin1_n
                        lower_maskB=self.B_data>=bin2; high_maskB=self.B_data<=bin2_n
                        final_maskA=np.logical_and(lower_maskA,high_maskA)
                        final_maskB=np.logical_and(lower_maskB,high_maskB)
                        final_maskAB=np.logical_and(final_maskA,final_maskB)
                        ##### _mask_count
                        mask_data=self.C_data[final_maskAB]
                        if len(mask_data.shape)==1:
                            mask_count=len(mask_data)
                        elif len(mask_data.shape)==2:
                            mask_count=mask_data.shape[0]*mask_data.shape[1]
                        elif len(mask_data.shape)==3:
                            mask_count=mask_data.shape[0]*mask_data.shape[1]*mask_data.shape[2]
                        ##### _apply_mask -> mask mean and std dev
                        #from IPython import embed; embed()
                        if mask_data.size==0:
                            mask_mean=0
                            mask_std=0
                        else:
                            mask_mean=mask_data.mean()
                            mask_std=mask_data.std()
                        self.final_count_3d.append([bin1,bin1_n,bin2,bin2_n,coord,mask_count,mask_mean,mask_std])
                        self.heat_map_count[coord[0]][coord[1]]=mask_count
                        self.heat_map_mean[coord[0]][coord[1]]=mask_mean
                        self.heat_map_std[coord[0]][coord[1]]=mask_std
                        #print([bin1,bin1_n,bin2,bin2_n,coord,mask_count,mask_mean,mask_std])
                                         
    def plot3d(self,mode="count",vmin=0,vmax=4095):
        count_array=np.array(self.final_count_3d)
        X=count_array[:,0]
        Y=count_array[:,2]
    
        if mode=="count":
            Z=count_array[:,4]
            vmax=Z.max()
            
        elif mode== "mean":
            Z=count_array[:,5]
        elif mode== "std":
            Z=count_array[:,6]
            vmax=Z.max()
        fig = plt.figure(figsize=(11, 8))
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(X, Y, Z, cmap=cm.gist_rainbow,vmin=vmin,vmax=vmax,
                       linewidth=0, antialiased=False,)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        
    def plot_heatmap(self,mode="count",minVal=0,maxVal=4096):
        cbarlabel=self.C_name+mode
        title=(self.A_name+' vs '+self.B_name+' - ',mode,' - ',self.C_name)
        if mode=="count":
            array=self.heat_map_count
            maxVal=array.max()
            cbarlabel="count"
            title=(self.A_name+' vs '+self.B_name+' - ',mode)
        elif mode== "mean":
            array=self.heat_map_mean
        elif mode== "std":
            array=self.heat_map_std
            maxVal=array.max()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(array,cmap='gist_rainbow',vmin=minVal,vmax = maxVal)
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)#, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(self.hist_binsB)))
        ax.set_yticks(np.arange(len(self.hist_binsA)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.hist_binsB)
        ax.set_yticklabels(self.hist_binsA)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
            # Loop over data dimensions and create text annotations.
        for num1,bin1 in enumerate(self.hist_binsA):
            for num2,bin2 in enumerate(self.hist_binsB):
                value=int(array[num1, num2])
                if value>10:
                    text = ax.text(num2, num1, value,
                                ha="center", va="center", color="b",fontsize=6)

        
        ax.set_title(title)
        fig.tight_layout()
        plt.show()    