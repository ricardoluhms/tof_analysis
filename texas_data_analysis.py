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
from read_write.utils import reader, check_dir
from label_crop import Crops


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

    def read(self,crop_mode=False,crop_count=4,label_mode=True):
        file_counter=0
        titles=["frame", "i_pxl", "j_pxl"]
        crop_coord=[]
        for file in self.pathlist:
            pack_counter=0
            pack = reader(file)
            dname,fname=file.split("\\")
            exp_depth=int(dname.split("/")[-1])/1000
            name,_=fname.split(".")
            titles.append(name)
            if file_counter==0:
                frame_i_j_pack=self._create_coord(pack.width,pack.height,pack.frames_count)
            for frame in range(pack.frames_count):
                _, array = pack.read()
                if crop_mode and name=="amplitude" and frame==1:
                    crops=Crops()
                    for c_count in range(crop_count):
                        if c_count==0:
                            roi_crop, roi_img=crops.crop_img(array.reshape((pack.height,pack.width)),
                                                             crop_obj=c_count, std_label=label_mode,exp_depth=exp_depth)      
                        else:
                            roi_crop, roi_img=crops.crop_img(roi_img,crop_obj=c_count,std_label=label_mode,exp_depth=exp_depth)
                        #roi_crop.append(depth)
                        #
                        crop_coord.append(roi_crop)
                if pack_counter==0:
                    array_pack=array
                else:
                    array_pack=np.vstack((array_pack,array))
                pack_counter+=1
            if file_counter==0:
                
                feature_pack=np.hstack((frame_i_j_pack,array_pack))
            else:
                
                feature_pack=np.hstack((feature_pack,array_pack))
            file_counter+=1
        
        df=pd.DataFrame(feature_pack,columns=titles)
        if crop_mode:
            return feature_pack,df,crop_coord
        else:
            return feature_pack,df,False
    
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
    
    def pack_data(self,crop_mode=True,crop_count=1):
        bbox_list=[]
        for num, experiment in enumerate(self.paths):
            _,exp_folder=experiment.split(self.all_exp_fld)
            exp_depth=int(exp_folder)/1000
            data=Out_texas_reader(experiment)
            array,df,crop_coord=data.read(crop_mode=crop_mode,crop_count=crop_count)
            if crop_coord!=False:
                bbox_list.append(crop_coord)
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
        
        return main_array, new_df, bbox_list

    @staticmethod
    def save_main_data_pack(df,file_output="D:/26022020/exp_resume.csv"):
        print("Start CSV save -  This may take a while")
        df.to_csv(file_output,index=False)
        print("Save complete")

    @staticmethod
    def load_main_data_pack(file_input="D:/26022020/exp_resume.csv"):
        print("Reading CSV save -  This may take a while")
        df=pd.read_csv(file_input)
        print("Load complete")
        return df

    @staticmethod
    def create_label_df(bbox_list,save=True,file_output="D:/26022020/crop_coord_label1.xlsx"):
        bbox_reshaped=[]
        for experiment in bbox_list:
            for data in experiment:
                bbox_reshaped.append(data)       
        dflabel=pd.DataFrame(bbox_reshaped,columns=["i_min","j_min","i_max","j_max","crop_label","exp_depth"])
        if save:        
            dflabel.to_excel(file_output)
        return dflabel

    @staticmethod
    def load_label_df(file_input="D:/26022020/crop_coord_label1.xlsx"):
        dflabel=pd.read_excel(file_input)
        dflabel=dflabel.drop(columns=["Unnamed: 0"])
        return dflabel

    @staticmethod
    def apply_label_df(dflabel,df):
        for crop in range(len(dflabel)):
            i_min,j_min,i_max,j_max,target_name,exp_depth=dflabel.iloc[crop]
            if1=df["i_pxl"]>=i_min; if2=df["i_pxl"]<=i_max
            jf1=df["j_pxl"]>=j_min; jf2=df["j_pxl"]<=j_max 
            mask_i=if1 & if2;  mask_j=jf1 & jf2;  mask_i_j=mask_i & mask_j
            mask_d=df["expected_depth"]==exp_depth
            mask_i_j_d=mask_i_j&mask_d
            df.loc[mask_i_j_d,"label"]=target_name
        return df
    
    @staticmethod
    def depth_plot(df,x_label="expected_depth",y_label="phase",labels_from_df=True,labels=["None"]):
        if labels_from_df:
            labels=df["label"].drop_duplicates().to_list()
        plot_list=[]
        ls = 'dotted'
        title_name=("expected_depth"+"  vs  "+y_label+" Mean+-Std")
        for lbnum,label in enumerate(labels):
            ttlb=len(labels)
            plot_num=ttlb*100+10+(lbnum+1)
            mask1=df["label"]==label
            x_values=df[x_label].drop_duplicates().to_list()

            if lbnum==0:
                ax1 = plt.subplot(plot_num)
                ax1.set_title(title_name+" - "+label)
            elif lbnum==1:
                ax2 = plt.subplot(plot_num, sharex=ax1)
                ax2.set_title(title_name+" - "+label)
            elif lbnum==2:
                ax3 = plt.subplot(plot_num, sharex=ax1)
                ax3.set_title(title_name+" - "+label)
            for x_value in x_values:
                mask2=df[x_label]==x_value
                mask=mask1 & mask2
                series=df[mask][y_label]
                if lbnum==0:
                    ax1.errorbar(x_value,series.mean(),yerr=series.std(),marker='o',linestyle=ls,label=label)
                    if y_label=="phase":
                        y1=df["expected_phase"].drop_duplicates()
                        ax1.plot(x_values, y1, 'o-')
                    if y_label=="depth":
                        y1=df["expected_depth"].drop_duplicates()
                        ax1.plot(x_values, y1, 'o-')
                elif lbnum==1:
                    ax2.errorbar(x_value,series.mean(),yerr=series.std(),marker='o',linestyle=ls,label=label)
                    if y_label=="phase":
                        y2=df["expected_phase"].drop_duplicates()
                        ax2.plot(x_values, y2, 'o-')
                    if y_label=="depth":
                        y1=df["expected_depth"].drop_duplicates()
                        ax1.plot(x_values, y1, 'o-')
                elif lbnum==2:
                    ax3.errorbar(x_value,series.mean(),yerr=series.std(),marker='o',linestyle=ls,label=label)

        #plt.legend()
        plt.show()
        #plot_array=np.array(plot_list)
                    
def main():
    ### Example Code
    ###### Amp Mask Value removes data which are below the threshold value
    all_exp_folder="D:/26022020/" #arquivo com os experimentos
    output_folder="D:/26022020/DATA_OUTPUT/" # arquivo com para saída de experimentos
    check_dir(output_folder)
    a=Exp_pack(all_exp_fld=all_exp_folder,amp_mask_value=20)
    ###### Evaluate multiple folders at once and define region for analysis (crop_count = number of regions to crop)
    array, df, bboxes=a.pack_data(crop_mode=True,crop_count=1)
    from IPython import embed; embed()
    ###### create df with labels based on the selected regions
    dflabel=a.create_label_df(bboxes,save=True,file_output=output_folder+"crop_coord_labelV0.xlsx")
    #### if dflabel exists it can be loaded
    #dflabel=a.load_label_df(file_input="D:/26022020/crop_coord_labelV0.xlsx")
    #### to apply the labels over the original dataset
    df=a.apply_label_df(dflabel,df)

    #### save the main df as a checkpoint, to avoid new image annotations 
    a.save_main_data_pack(df,file_output=output_folder+"example_exp_resume.csv")
    #### load a main df checkpoint 
    df=a.load_main_data_pack(file_input=output_folder+"filtered_exp_resume.csv")
    #### add calculated values
    df["phase error"]=abs(df["expected_phase"]-df["phase"])
    df["depth error"]=abs(df["calculated_depth"]-df["expected_depth"])

    a.depth_plot(df) ### x_label="expected_depth" ; y_label="phase"
    a.depth_plot(df,y_label="amplitude")
    a.depth_plot(df,y_label="phase error")
    a.depth_plot(df,y_label="depth error")
    from IPython import embed;embed()


if __name__ == '__main__':
    main()