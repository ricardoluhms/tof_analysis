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
#from samples.texas.read_recorded.main import gui
from read_write.utils import reader

class Main_folder_select():
    def __init__ (self,all_exp_folder):
        self.all_exp_folder=all_exp_folder
        
    def swipe_folders(self):
        self.folder_path_list=[]
        for folder in os.listdir(self.all_exp_folder):
            n_f_path=os.path.join(self.all_exp_folder,folder)       
            if os.path.isdir(n_f_path): 
                if n_f_path not in self.folder_path_list:
                    self.folder_path_list.append(n_f_path)

class Bin_texas_reader():
    def __init__(self,input_fld):
        #Texas Tof Model ['OPT8241']
        self.input_folder=input_fld
        self.device_resol=[240,320] #### Adjust according to Kit Resolution - Standard is Texas Kit
        self.tt_points_per_frame=self.device_resol[0]*self.device_resol[1]
        self.path_list=[]
        self.file_dict={}
        self._file_type_dict()
        self.file_list(show_file=False)
        self.rename_amplitude()

    def _file_type_dict(self):
        file_type_List=['Ambient',
        '0Amplitude','AmplitudeAvg','AmplitudeStd',
        'Depth','DepthAvg','DepthStd','Distance',
        'Phase','PhaseAvg','PhaseStd',
        'PointCloud']

        dtype_list=[np.uint8,
        np.uint16,np.uint16,np.uint16,
        np.float32,np.float32,np.float32,np.float32,
        np.uint16,np.uint16,np.uint16,
        np.float32]
        
        for i in range(len(file_type_List)):
            filedetails={}
            filetype=file_type_List[i]
            dtype=dtype_list[i]
            filedetails['dtype']=dtype
            if filetype!='PointCloud' and filetype in file_type_List:
                filedetails['map']=1
            elif filetype=='PointCloud':
                filedetails['map']=4
            self.file_dict[filetype]=filedetails
            
    def file_list(self,show_file=False):
        valid_files = ".bin"
        for file in os.listdir(self.input_folder):
            if file.endswith(valid_files):
                n_path=os.path.join(self.input_folder,file)
                if n_path not in self.path_list:
                    self.path_list.append(n_path)
                    
    def rename_amplitude(self):
        valid_files = ".bin"
        for file in os.listdir(self.input_folder):
            if (file.endswith(valid_files)and file.split('.')[0]=="Amplitude"):
                old=self.input_folder+'/'+file
                new=self.input_folder+'/0'+file
                os.rename(old, new)                    

class Bin_to_array():
    def __init__(self,device_resol,tt_points_per_frame,file_dict):
        self.device_resol=device_resol
        self.tt_points_per_frame=tt_points_per_frame
        self.file_dict=file_dict
        
    def _reshape_data(self,data,filetype,fmap):
        # Reshaping data
        if fmap==1:
            frames=int(len(data)/self.tt_points_per_frame)
            data=data.reshape(frames,self.device_resol[1],self.device_resol[0],fmap).swapaxes(-1,-3)
        elif fmap==4:
            frames=int(len(data)/(self.tt_points_per_frame*fmap))
            data=data.reshape(frames,fmap,self.device_resol[0],self.device_resol[1])
        return data
    
    def _raw_single_file(self,file):
        _,b=os.path.split(file)
        filetype,_=os.path.splitext(b)
        #print("file name =",file," filetype =",filetype)
        if filetype in self.file_dict:
            dtype=self.file_dict[filetype]['dtype']
            fmap=self.file_dict[filetype]['map']
            data=np.fromfile(file,dtype=dtype)
        return data,filetype,fmap
    
    def reshape_single(self,file):
        data,filetype,fmap=self._raw_single_file(file)
        reshaped_data=self._reshape_data(data,filetype,fmap)
        return reshaped_data,filetype,fmap
    
    def reshape_multi(self,path_list):
        count=0
        for file in path_list:
            data,filetype,fmap=self._raw_single_file(file)
            reshaped_data=self._reshape_data(data,filetype,fmap)
            if count==0:
                dataGroup=reshaped_data
            else:
                dataGroup=np.hstack((dataGroup,reshaped_data))
            count+=1
        return dataGroup

class Array_to_dataframe():
    def __init__(self,titles=["frame", "i_pxl", "j_pxl", "Amp",  "Amb", "Depth", "Dist",
                              "Phase",  "PC_x", "PC_y", "PC_z","PC_i"]):
        self.titles = titles
    
    @staticmethod
    def _create_coord(x,y,frames):
        ### p1
        i_pxl=(np.arange(0,x)).reshape(x,1)
        ### p2
        for j in range(y):
            j_pxl=(np.ones(i_pxl.shape[0])*j).reshape(i_pxl.shape[0],1)
            i_j_ct=np.hstack((i_pxl,j_pxl))
            if j==0:
                i_j_pack=i_j_ct
            else:
                i_j_pack=np.vstack((i_j_pack,i_j_ct))
        ### p3
        for frame in range(frames):
            frame_ct=(np.ones(i_j_pack.shape[0])*frame).reshape(i_j_pack.shape[0],1)
            if frame==0:
                frame_i_j_pack=np.hstack((frame_ct,i_j_pack))
            else:
                frame_i_j_pack=np.vstack((frame_i_j_pack,np.hstack((frame_ct,i_j_pack))))
        frame_i_j_pack=frame_i_j_pack.astype("int")
        return frame_i_j_pack
    
    def _rearrage_test(self,array):
        frames,features,y,x =array.shape
        for frame in range(frames):
            a=array[frame]
            for fmap in range(features):
                b=a[fmap].reshape((x*y),1)
                if fmap==0:
                    feat_stack=b
                else:
                    feat_stack=np.hstack((feat_stack,b))
            if frame==0:
                frame_feat_stack=feat_stack
            else:
                frame_feat_stack=np.vstack((frame_feat_stack,feat_stack))
        return frame_feat_stack
       
    # Step 1 - Get Array from Raw data - bin files
    def reshaped_into_table(self,array, mx_frames=0):
        mx_frames

        print("Initial shape = ", array.shape)
        if mx_frames>0:
            array=array[:mx_frames,:,:,:]
            print("Frame limited shape = ", array.shape)
        frames,_,y,x=array.shape
        
        array_table=self._rearrage_test(array)
        print("Rearranged shape = ", array_table.shape)
        coord=self._create_coord(x,y,frames)
        array_table=np.hstack((coord,array_table))
        print("Conversion from Array into Frames")
        print("Assigned pixel coordinates and frames for each bin file ", array_table.shape)
        return array_table

class Out_texas_reader(Array_to_dataframe,Main_folder_select):
    def __init__(self,input_fld):
        self.pathlist=[]
        for file in os.listdir(input_fld):
            path=os.path.join(input_fld,file)
            self.pathlist.append(path)

    def read(self):
        file_counter=0
        titles=["frame", "i_pxl", "j_pxl"]
        
        for file in self.pathlist:
            pack_counter=0
            pack = reader(file)
            #from IPython import embed;embed()
            _,fname=file.split("\\")
            name,_=fname.split(".")
            titles.append(name)
            if file_counter==0:
                frame_i_j_pack=self._create_coord(pack.width,pack.height,pack.frames_count)

            for frame in range(pack.frames_count):
                _, array = pack.read()
                #array=array.reshape((pack.height,pack.width))
                if pack_counter==0:
                    array_pack=array 
                else:
                    array_pack=np.vstack((array_pack,array))
                pack_counter+=1
            if file_counter==0:
                feature_pack=np.hstack((frame_i_j_pack,array_pack))
            else:
                #from IPython import embed;embed()
                feature_pack=np.hstack((feature_pack,array_pack))
            file_counter+=1
        
        df=pd.DataFrame(feature_pack,columns=titles)
        return feature_pack,df
    
    @staticmethod
    def depth2phase(freq1=16,freq2=24,ph_mask=2,exp_depth=1000,phase_corr=0):
        dual_freq=np.gcd(freq1,freq2)
        ma=freq1/dual_freq
        mb=freq2/dual_freq
        l_speed=299792458 #m/s
        max_range=l_speed/(2e6*dual_freq)
        calc_phase=np.round((ma*mb*(2**12))/(max_range*(2**(5-ph_mask)))*exp_depth,decimals=0)
        return calc_phase

    @staticmethod
    def phase2depth(freq1=16,freq2=24,ph_mask=2,phase_in=0,phase_corr=0):
        dual_freq=np.gcd(freq1,freq2)
        ma=freq1/dual_freq
        mb=freq2/dual_freq
        l_speed=299792458 #m/s
        max_range=l_speed/(2e6*dual_freq)
        calc_depth=np.round(phase_in*(max_range*(2**(5-ph_mask)))/(ma*mb*(2**12)),decimals=4)
        return calc_depth

class Exp_pack(Out_texas_reader):
    def __init__(self,all_exp_fld="D:/26022020/",amp_mask_value=20):
        mfld=Main_folder_select(all_exp_fld)
        mfld.swipe_folders()
        self.paths=mfld.folder_path_list
        self.all_exp_fld=all_exp_fld
        self.amp_mask_value=amp_mask_value
    
    def pack_data(self):

        for num, experiment in enumerate(self.paths):
            _,exp_folder=experiment.split(self.all_exp_fld)
            exp_depth=int(exp_folder)/1000
            data=Out_texas_reader(experiment)
            array,df=data.read()
            amp_mask=array[:,4]>self.amp_mask_value
            masked_array=array[amp_mask]
            expected_exp_phase=data.depth2phase(exp_depth=exp_depth)
            expected_exp_depth_array=np.ones((masked_array.shape[0],1))*exp_depth

            expected_exp_phase_array=np.ones((masked_array.shape[0],1))*expected_exp_phase
            calc_depth_array=data.phase2depth(phase_in=masked_array[:,6])
            calc_depth_array=calc_depth_array.reshape(len(calc_depth_array),1)
            pack_array=np.hstack((masked_array,
                                  expected_exp_phase_array,
                                  expected_exp_depth_array,
                                  calc_depth_array))
            if num==0:
                main_array=pack_array
            else:
                main_array=np.vstack((main_array,pack_array))
        old_columns=df.columns.to_list()
        n_columns=['expected_phase', 'expected_depth', 'calculated_depth']
        n_columns=old_columns+n_columns
        new_df=pd.DataFrame(main_array,columns=n_columns)
        new_df["label"]="None"
        return main_array, new_df

    def bounding_box_label(self,df,i_min,i_max,j_min,j_max,target_name,exp_depth):
        mask_i=(df["i_pxl"]>=i_min & df["i_pxl"]<=i_max)
        mask_j=(df["j_pxl"]>=j_min & df["j_pxl"]<=j_max)
        mask_i_j=mask_i&mask_j
        mask_d=df["expected_depth"]==exp_depth
        mask_i_j_d=mask_i_j&mask_d
        df.loc[mask_i_j_d,"label"]=target_name
        return df

    def df_with_label(self,boxbox_list,df):
        for bbox in boxbox_list:
            i_min,i_max,j_min,j_max,target_name,exp_depth=bbox
            df=self.bounding_box_label(df,i_min,i_max,j_min,j_max,target_name,exp_depth)
        return df

    def depth_plot(self,df,y_label="phase",labels=["None"]):
        x_label="expected_depth"
        plot_list=[]
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)
        ls = 'dotted'
        title_name1=("expected_depth"+" vs "+y_label+" Mean+-Std")
        title_name2=("expected_depth"+"event count")
        ax1.set_title(title_name1)
        ax2.set_title(title_name2)

        for label in labels:
            mask=df["label"]==label
            x_values=df[x_label].drop_duplicates().to_list()
            for x_value in x_values:
                mask=df[x_label]==x_value
                series=df[mask][y_label]
                ax1.errorbar(x_value,series.mean(),yerr=series.std(),marker='o',linestyle=ls,label=label)
                ax2.plot(x_value,mask.sum(),marker='o',linestyle=ls,label=label)
                #plot_list.append([x_value,series.mean(),series.std(),mask.sum(),label])
        plt.show()
        #plot_array=np.array(plot_list)
                    


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

def main():
    #exp_dir="D:/26022020/"
    #a.loop()
    all_exp_folder="D:/26022020/"
    a=Exp_pack(all_exp_fld=all_exp_folder)
    array,df=a.pack_data()
    a.depth_plot(df)
    df["phase error"]=abs(df["expected_phase"]-df["phase"])
    df["depth error"]=abs(df["calculated_depth"]-df["expected_depth"])
    mask1=df["phase error"]<=50
    df2=df[mask1]
    a.depth_plot(df2)
    a.depth_plot(df2,y_label="amplitude")
    a.depth_plot(df2,y_label="phase error")
    a.depth_plot(df2,y_label="depth error")
    # mfld=Main_folder_select(all_exp_folder)
    # mfld.swipe_folders()
    # mfld.all_exp_folder
    # a=mfld.folder_path_list[0]
    # b=Out_texas_reader(a)
    # arraypack,df=b.read()
    
    # binf=Bin_texas_reader(a)
    # binf.path_list
    # binf.device_resol
    # binf.path_list[0]
    # bin_arr=Bin_to_array(binf.device_resol,binf.tt_points_per_frame,binf.file_dict)
    # data,filetype,fmap=bin_arr.reshape_single(binf.path_list[0])
    # dataG=bin_arr.reshape_multi(binf.path_list)

    #ambient_resume=arraypack[:,3]
    
    # amp_mask=arraypack[:,4]>20
    # masked_pack=arraypack[amp_mask]
    from IPython import embed; embed()

    #flags_resume=arraypack[:,5]
    #phase_resume=arraypack[:,6]
    # #################################################################################################
    # A_data=dataG[:,0,:,:]
    # A_data=masked_pack[:,4]

    # ganim=Graph_Anim(A_data,step=100,axis_name=" ",lower_thresh=100)
    # ganim.start_graph()
    # ganim.update_graph()
    # plt.show()

    # mask=dataG[:,0,:,:]>0
    # maskAmb1=dataG[:,1,:,:]>0.9
    # maskAmb2=dataG[:,1,:,:]<13.1
    # mask1=np.logical_and(maskAmb1,maskAmb2)
    # mask=np.logical_and(mask,mask1)
    # A_data=dataG[:,0,:,:][mask]
    # B_data=dataG[:,1,:,:][mask]
    # C_data=(dataG[:,4,:,:]-1230)[mask]
    
    # B_pack,A_pack,C_pack=[B_data,"Ambient",1],[A_data,"Amplitude",200],[C_data,"Phase",100]
    # b1=Triple_Param_Analysis(A_pack,B_pack,C_pack)
    # b1.param_calc()
    # # #b1.plot3d(final_count,mode="mean")

    # b1.plot_heatmap(mode="count")
    # b1.plot_heatmap(mode="mean")
    # b1.plot_heatmap(mode="std")
    # # #################################################################################################
    # mask=dataG[:,0,:,:]>19 ### Amplitude Mask 0
    # mask1=dataG[:,1,:,:][mask]>0.99 ### Ambient Mask 1
    # mask2=dataG[:,0,:,:][mask][mask1]<1200 ### Amplitude Mask 2

    # D_data=dataG[:,1,:,:][mask][mask1][mask2]
    # A_data=dataG[:,0,:,:][mask][mask1][mask2]
    # a1=Dual_Param_Analysis(A_data,D_data,"Amplitude","Ambient",stepA=20,stepB=1)
    # count_map1=a1.loop_mask()
    # a1.plot(count_map1)

if __name__ == '__main__':
    main()