import numpy as np
import cv2
import pandas as pd
#### Folder and data libraries
import os
#### Plot Libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as ptc

#### Customized Libraries - Luhm
from std_headers import Headers
from input_data import Input_data, Mfolder
from feature_selection import Feature_selection
from exp_data import Exp_data
#### Customized Libraries  - Vinicius
from custom_modules.filters.utils import filter

#####
"""
Objectives:
-- Add Multiple Crop Feature
-- Modify Output file crop multiple crop names for a single experiment.
-- Adjust Errormap to group results by label and experiments 
"""
def main():
    ### File path to load data of all experiment setup data loading
    file_path="C:/Users/ricar/Downloads/Experimento_TOF_Texas.xlsx"
    ### folder path to load data of a single TOF experiment 
    #folder_path='D:/Codes/Videos_tof/Exprimentos_atenuadores/Exp004'
    ### File path to load data of all TOF exesetup datperiment a loading
    all_folder_path='D:/Codes/Videos_tof/Experimentos_atenuadores/'
    da=Data_analysis(all_folder_path,file_path)
    crop_activate_list=[0,3,8,11,16,19]
    #crop_activate_list=[0,5,10]
    da.select_crop_folders(crop_activate_list)
    ### Select Number of Crops
    #output="d:/tof/output8.xlsx"
    final_count_pack=da.run_data_analysis(folders_numb=24,frame_limit=30)


    depth_list=["1013","3009","4834"]
    aten_list=["Sem filtro","Fume 1","Fume 2",
               "Pelicula 1","Pelicula 2","Pelicula 3","Pelicula 4","Pelicula 5"]
    
    for depth in depth_list:
        lst=[depth]
        for aten in aten_list:
            lst2=[aten]
            aten_pack=Data_analysis.filter_by_atenuator(final_count_pack,aten_list=lst2)
            depth_pack=Data_analysis.filter_by_depth(aten_pack,depth_list=lst)
            Data_analysis.error_plot(depth_pack)

    from IPython import embed;embed()
    #columns=['atenuador','exp_depth','amp_mean','amp_std','error_mean','error_std'],
    #filename_export="d:/tof/output.xlsx"
class Data_analysis():
    def __init__(self,all_exp_folder,exp_setup_filepath):
        #self.file_path=exp_setup_filepath
        #self.main_folder=all_exp_folder
        self.mfld=Mfolder(all_exp_folder)
        self.mfld.swipe_folders()
        self.edt=Exp_data(exp_setup_filepath)
        self.resumed_exp_data=[]
        self.detailed_exp_data=[]

    def select_crop_folders(self,crop_folder_list):
        self.crop_activate_list=crop_folder_list

    def single_folder(self,folder_count,frame_limit):
        folder_path=self.mfld.folder_path_list[folder_count]
        self.edt.get_exp_numb(folder_path)
        self.exp_depth=self.edt.get_exp_data(self.edt.exp_numb,"dist_mm")
        self.exp_aten=self.edt.get_exp_data(self.edt.exp_numb,"Atenuador")
        self.idt=Input_data(folder_path)
        self.grouped_array=self.idt.reshaped_grouped()[:frame_limit,:,:,]

    def activate_crop(self,folder_count,
                      frame_num=0,feature_num=0,
                      crop_mode="single",crop_object_num=1,std_label=False):    
        #frame_num=0 #Uses the first frame
        #feature_num=0 where 0 = Amplitude      
        if folder_count in self.crop_activate_list:
            img_o, self.bin_mask=self.fs.get_otsu(self.grouped_array,frame_num,feature_num)
            if crop_mode=="single":
                #####
                #crop_label=self.cp.add_label(std_label=std_label)
                self.roi_crop=self.cp.crop_img(img_o,self.exp_aten,std_label)
                self.roi_crop_pack=self.roi_crop
            if crop_mode=="multiple":
                for crop_obj in range(crop_object_num):
                    #crop_label=self.cp.add_label(label_num=crop_obj,std_label=std_label)
                    self.roi_crop=self.cp.crop_img(img_o,self.exp_aten,std_label,crop_obj)
                    self.roi_crop_pack.append(self.roi_crop)

    def run_data_analysis(self,
                          folders_numb=15, frame_limit=30,
                          crop_mode="single" ):
        #### Available crop_mode = ["single","multiple"]
        final_count_pack=[]
        self.roi_crop_pack=[]
        for folder_count in range(folders_numb):
            self.single_folder(folder_count,frame_limit)
            self.fs=Feature_show()
            self.cp=Crop()
            self.activate_crop(
                               folder_count,
                               frame_num=0,feature_num=0,
                               crop_mode=crop_mode,crop_object_num=1,
                               std_label=True
                               )

            if crop_mode=="single":
                error_depth_array,eda_m,amp_data,apa_m=self.single_object_analysis(self.roi_crop_pack)
                
                self.resumed_exp_data.append(( self.exp_aten, self.exp_depth,
                                                apa_m.mean(), apa_m.std(),
                                                eda_m.mean(), eda_m.std() ))

                final_count,name=self._apply_historgram(amp_data,error_depth_array)
                final_count_pack.append((final_count,name,self.roi_crop_pack[0][1]))
                #final count list format [amplitude,mask_count,mask_mean,mask_std]

            if crop_mode=="multiple":
                pass
                #for obj_crop in range(crop_object_num):
                    

                #from IPython import embed;embed()
        return final_count_pack

            ###################################################
            #from IPython import embed; embed()
            
        #columns=['atenuador','exp_depth','amp_mean','amp_std','error_mean','error_std']
        #self.df=pd.DataFrame(self.resumed_exp_data,columns=columns)
        #self.df.to_excel(filename_export)

    def single_object_analysis(self,roi_crop_pack):
        # self.cp=Crop()
        # self.fs=Feature_show())
        cropped_array=self.cp.multi_feature_crop(self.grouped_array,roi_crop_pack)
        cropped_mask=self.cp.single_feature_crop(self.bin_mask,roi_crop_pack)
        cropped_masked_array=self.fs.apply_mask(cropped_array,cropped_mask)
        error_depth_array, error_mask=self.fs.apply_depth_check(cropped_masked_array,self.exp_depth)
        #### Check Effects of error mask over depth array and amp_data
        eda_m=error_depth_array[error_mask]
        amp_data=cropped_masked_array[:,0,:,:]
        apa_m=amp_data[error_mask]
        return error_depth_array,eda_m,amp_data,apa_m

    def _apply_historgram(self,amp_data,error_depth_array,plot=False):
        self.amp_h=Main_Histogram(amp_data)
        self.amp_h.histo()
        self.err_d=Depth_Error_hist(self.amp_h.hist_bins)
        final_count=self.err_d.loop_mask(amp_data,error_depth_array)
        ## final_count = list with [amplitude_pair[1],mask_count,mask_mean,mask_std]
        if len(str(self.edt.exp_numb))==1:
            e_name="Exp_00"
        elif len(str(self.edt.exp_numb))==2:
            e_name="Exp_0"
        else:
            e_name="Exp_"
        name=e_name+str(self.edt.exp_numb)+" - Aten: "+self.exp_aten+" - Dist: "+str(self.exp_depth)
        if plot==True:
            self.err_d.plot(name,final_count)
        return final_count,name

    def read_saved_analysis(self,excel_filename):
        self.df=pd.read_excel(excel_filename, index_col=0)
        self.dfr=Dataframe_results(self.df)
        print("Available Column Names",self.dfr.label_names)
    
    @staticmethod
    def error_plot(count_pack):
        jitter=0.5
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Amplitude vs Depth_Error Mean+-Std")         
        ls = 'dotted'
        for exp_num,experiment in enumerate(count_pack):

            data=np.array(experiment[0])
            if len(data)!=0:
                exp_name=experiment[1]
                label=experiment[2]
                label_name=exp_name+"_"+label
                x_axis_values=data[:,0]+exp_num*jitter
                y_mean=data[:,2]
                y_std=data[:,3]

                ax.errorbar(x_axis_values,
                                        y_mean,
                                    yerr=y_std,
                                    linestyle=ls,
                                    marker="o",
                                    label=label_name)
                # name1=exp_name.split(" - ")[0]
                # name2=exp_name.split("Aten: ")[1].split(" - ")[0]
                # if len(name2.split(" "))>1:
                #     name2=name2.split(" ")[0]+"_"+name2.split(" ")[1]
                # name3=exp_name.split("Dist: ")[1]
                # name=name1+"_"+name2+"_"+name3       
                # ax.legend(loc='lower right')
                # file_name="C:/Users/ricar/Desktop/Output/"+name+".jpg"
                # plt.savefig(file_name)
        plt.show()
        
    def plot_setup(self,x_axis_col_name='exp_depth',
                        y_mean_column='amp_mean',
                        y_std_column='amp_std',
                        label='atenuador'):

        self.dfr.set_x_axis(x_axis_col_name)
        self.dfr.set_y_axis(y_mean_column,y_std_column)
        self.dfr.set_label(label)
    
    @staticmethod
    def filter_by_label(final_count_pack,label_list):
        filtered_pack=[]
        for experiment in final_count_pack:
            data=np.array(experiment[0])
            if len(data)!=0:
                label=experiment[2]
                if label in label_list:
                    filtered_pack.append(experiment)
        return filtered_pack

    @staticmethod
    def filter_by_depth(final_count_pack,depth_list):
        filtered_pack=[]
        for experiment in final_count_pack:
            data=np.array(experiment[0])
            if len(data)!=0:
                exp_name=experiment[1]
                depth=exp_name.split("Dist: ")[1]
                if depth in depth_list:
                    filtered_pack.append(experiment)
        return filtered_pack
        
    @staticmethod
    def filter_by_atenuator(final_count_pack,aten_list):
        filtered_pack=[]
        for experiment in final_count_pack:
            data=np.array(experiment[0])
            if len(data)!=0:
                exp_name=experiment[1]
                aten=exp_name.split("Aten: ")[1].split(" -")[0]
                if aten in aten_list:
                    filtered_pack.append(experiment)
        return filtered_pack

class Feature_show():
    def __init__(self):
        pass

    @staticmethod
    def std_window_show(window_name,array):
        cv2.namedWindow(window_name,cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(window_name,array)

    @staticmethod
    def _check_exploding(img):
        exp_threshold=img.mean()+3*img.std()
        mask=img[:,:]>exp_threshold
        sum_mask=mask.sum()
        if sum_mask>1:
            img[mask]=exp_threshold#+img.std()
        return img

    def feature_show(self,grouped_array):
        for i in range(grouped_array.shape[0]):
            self.std_window_show("Amplitude",filter.norm(self._check_exploding(grouped_array[i][0])))
            self.std_window_show("Ambient",filter.norm(self._check_exploding(grouped_array[i][1])))
            self.std_window_show("Depth",filter.norm(self._check_exploding(grouped_array[i][2])))
            #grouped_array[i][2]))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_otsu(self,grouped_array,frame_num,feature_num):
        frame_img=grouped_array[frame_num][feature_num]
        ##########################################################
        img=filter.norm(self._check_exploding(frame_img))
        mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        mask=(mask==255)
        img=img*mask
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        return img, mask

    def apply_depth_check(self,any_grouped_array,exp_depth,plot_show=False):

        error_depth_array=any_grouped_array[:,2,:,:]-exp_depth/1000
        eda=error_depth_array
        #print(eda.shape[0]*eda.shape[1]*eda.shape[2])
        #print(exp_depth/1000)
        eda=error_depth_array.copy()
        eda_mask=abs(eda[:,:,:])<0.08
        #print(eda_mask.sum())
        if plot_show==True:
            for i in range(eda.shape[0]):
                ###### Code under development - 
                img=eda[i]
                img=filter.norm(img)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                self.std_window_show("Depth check",img)
                cv2.waitKey()
                cv2.destroyWindow("Depth check")
            
        return error_depth_array, eda_mask

    def apply_mask(self,any_grouped_array,mask,mask_type="binary"):
        frames,features,_,_=any_grouped_array.shape
        masked_array=any_grouped_array.copy()
        if mask_type!="binary":
            mask=(mask==255).reshape((240,320))
        for frame in range(frames):
            for feature in range(features):
                masked_array[frame,feature:,:]=any_grouped_array[frame,feature:,:]*mask
        return masked_array

class Crop(Feature_show):

    def click_and_crop(self,event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        global refPt, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True  # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that the cropping operation is finished
            refPt.append((x, y))
            cropping = False
        if len(refPt)==2:
            cv2.rectangle(img, refPt[0], refPt[1], (0,255,255), 2)
            roi, _=self._crop_coord_detect(refPt,img)
            self.std_window_show('Crop',roi)
    
    def crop_img(self,img_o,exp_aten,std_label,crop_obj=1):
        global img, refPt, cropping
        refPt=[]
        roi_crop=[]
        cropping = False
        img = img_o.copy()
        img_clone = img.copy()
        cv2.namedWindow('Image',cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('Image', self.click_and_crop)
        print('Select the desired area to crop')
        while True:
            skip_crop=False
            cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF   
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                print("r key pressed")
                img = img_clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                if len(refPt) < 2:
                    skip_crop=True
                else:
                    cv2.destroyWindow('Crop')
                    break              

        if (len(refPt) == 2 and skip_crop==False):
            _,roi_coord = self._crop_coord_detect(refPt,img_clone)
            ## roi_coord format output= [x_min,y_min,x_max,y_max]
        
        cv2.destroyAllWindows()
        crop_label=self.add_label(label_num=crop_obj,std_label=std_label)
        roi_crop.append((roi_coord,crop_label,exp_aten))
        return roi_crop

    def _crop_coord_detect(self,refPt,img):
        x_max=max(refPt[0][0],refPt[1][0])
        x_min=min(refPt[0][0],refPt[1][0])
        y_max=max(refPt[0][1],refPt[1][1])
        y_min=min(refPt[0][1],refPt[1][1])
        roi = img[y_min:y_max, x_min:x_max]
        roi_coord = [x_min,y_min,x_max,y_max]
        return roi, roi_coord

    def add_label(self,label_num=1,std_label=True):
        if std_label==False:
            label=input("Set Label Name for cropped region:")
        else:
            label="std_label-"+str(label_num)
        return label

    def multi_feature_crop(self,grouped_array,roi_crop):
        if len(roi_crop)==1:
            roi_coord=roi_crop[0][0]
            # [x_min,y_min,x_max,y_max]
            group_crop=grouped_array[:,:,roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
        elif len(roi_crop)<1:
            print("Error - Crop cannot be performed without ROI coordinates")
            print("roi_crop input value = ",roi_crop)
            group_crop="error"
        return group_crop

    def single_feature_crop(self,simple_array,roi_crop):
        if len(roi_crop)==1:
            roi_coord=roi_crop[0][0]
            # [x_min,y_min,x_max,y_max]
            simple_crop=simple_array[roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
        elif len(roi_crop)<1:
            print("Error - Crop cannot be performed without ROI coordinates")
            print("roi_crop input value = ",roi_crop)
            simple_crop="error"
        return simple_crop

class Dataframe_results():
    def __init__(self,dataframe):
        self.df=dataframe

    def set_x_axis(self,x_axis_col_name):
        self.x_axis_name=x_axis_col_name
        print(self.x_axis_name)
        self.x_axis_values=self.get_unique(x_axis_col_name)
        print(self.x_axis_values)

    def set_y_axis(self,y_mean_column,y_std_column):
        self.y_mean_column=y_mean_column
        self.y_std_column=y_std_column
    
    def set_label(self,label_column):
        self.label_column=label_column
        self.label_names=self.get_unique(label_column)

    def get_unique(self,column_name):
        print(column_name)
        unique_values=self.df[column_name].drop_duplicates().values
        return unique_values
        
    def mean_std_values(self,label_name):
        df_filter=self.df[self.df[self.label_column]==label_name]
        y_mean_values=df_filter[self.y_mean_column].values
        y_std_values=df_filter[self.y_std_column].values
        #y_std_values=self.df[self.df[self.label_column]==label_name].ysc.values
        return y_mean_values,y_std_values

    def plot_all(self):

        _, ax = plt.subplots(figsize=(7, 4))

        #ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)
        ls = 'dotted'
        for label_name in self.label_names:
            y_mean,y_std=self.mean_std_values(label_name)
            
            ax.errorbar(self.x_axis_values,
                                    y_mean,
                                yerr=y_std,
                                linestyle=ls, 
                                marker='o',
                                label=label_name)
        ax.legend(loc='lower right')
        plt.show()

    def plot_single(self):
        label_name=input('Insert Label Name: ')
        _, ax = plt.subplots(figsize=(7, 4))

        #ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)
        ls = 'dotted'
        y_mean,y_std=self.mean_std_values(label_name)

        ax.errorbar(self.x_axis_values,
                                y_mean,
                            yerr=y_std,
                            linestyle=ls,label=label_name)

        ax.legend(loc='lower right')
        plt.show()

class Main_Histogram():
    def __init__(self,data):
        self.hist_min=data.min()
        self.hist_max=data.max()
        self.hist_bins=np.arange(self.hist_min,self.hist_max,10)
        self.data=data

    def histo(self):
        self.count, self.hist_bins=np.histogram(self.data,bins=self.hist_bins,density=False)

    def plot(self):
        if self.hist_min==0:
            _ = plt.hist(self.count[1:], bins=self.hist_bins[1:])
        else:
            _ = plt.hist(self.count, bins=self.hist_bins)
        plt.title("Histogram")
        plt.show()

class Depth_Error_hist(Main_Histogram):
    def __init__(self,bins):
        self.bin_pair=[]
        for i in range(len(bins)):
            if i+1<len(bins):
                self.bin_pair.append([bins[i],bins[i+1]])

    def _bins_mask(self,amp_data,bin_min,bin_max):
        lower_mask=amp_data[:,:,:]>=bin_min
        high_mask=amp_data[:,:,:]<=bin_max
        final_mask=np.logical_and(lower_mask,high_mask)
        return final_mask
    
    def _mask_count(self,final_mask,error_data):
        mask_data=error_data[final_mask]
        if len(mask_data.shape)==1:
            mask_count=len(mask_data)
        elif len(mask_data.shape)==2:
            mask_count=mask_data.shape[0]*mask_data.shape[1]
        elif len(mask_data.shape)==3:
            mask_count=mask_data.shape[0]*mask_data.shape[1]*mask_data.shape[2]
        return mask_count

    def _apply_mask(self,final_mask,error_data):
        mask_data=error_data[final_mask]
        mask_mean=mask_data.mean()
        mask_std=mask_data.std()
        return mask_mean,mask_std   

    def loop_mask(self,amp_data,error_data):
        final_count=[]
        for pair in self.bin_pair:
            final_mask=self._bins_mask(amp_data,pair[0],pair[1])
            mask_count=self._mask_count(final_mask,error_data)
            if mask_count!=0:
                mask_mean,mask_std=self._apply_mask(final_mask,error_data)
                final_count.append([pair[1],mask_count,mask_mean,mask_std])
        return final_count
            
    def plot(self,name,final_count):
        fct=np.array(final_count)
        x=fct[:,0]
        y=fct[:,2]
        yerr=fct[:,3]
        _, ax = plt.subplots(figsize=(7, 4))
        ls = 'dotted'
        ax.set_title("Amplitude vs Depth_Error Mean+-Std"+name)
        #ax.text(0,0,name, va="top", ha="left")
        for i in range(len(fct)):
            ax.errorbar(x,y,yerr=yerr,
                            marker='o',
                            linestyle=ls#,
                            #label=name
                        )
            #ax.legend(loc='lower right')
            #+"-Exp: "+name
        plt.show()
        
if __name__ == "__main__":
    main()
    pass