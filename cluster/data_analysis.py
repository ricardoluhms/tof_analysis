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
    from IPython import embed; embed()
    crop_activate_list=[0,5,10]
    da.select_crop_folders(crop_activate_list)
    output="d:/tof/output8.xlsx"
    da.run_data_analysis(folders_numb=15,frame_limit=30,filename_export=output)
    da.plot_setup()

class Data_analysis():
    def __init__(self,all_exp_folder,exp_setup_filepath):
        #self.file_path=exp_setup_filepath
        #self.main_folder=all_exp_folder
        self.mfld=Mfolder(all_exp_folder)
        self.mfld.swipe_folders()
        self.edt=Exp_data(exp_setup_filepath)
        self.result_table_list=[]

    def select_crop_folders(self,crop_folder_list):
        self.crop_activate_list=crop_folder_list

    def run_data_analysis(self,
                          folders_numb=15, frame_limit=30,
                          columns=['atenuador','exp_depth','amp_mean','amp_std','error_mean','error_std'],
                        filename_export="d:/tof/output.xlsx"):

        for folder_count in range(folders_numb):
            folder_path=self.mfld.folder_path_list[folder_count]
            self.edt.get_exp_numb(folder_path)
            exp_depth=self.edt.get_exp_data(self.edt.exp_numb,"dist_mm")
            exp_aten=self.edt.get_exp_data(self.edt.exp_numb,"Atenuador")
            self.idt=Input_data(folder_path)
            grouped_array=self.idt.reshaped_grouped()[:frame_limit,:,:,]
            self.gs=Feature_show()
            self.cp=Crop()
            if folder_count in self.crop_activate_list:
                roi_crop,bin_mask=self.cp.crop_otsu_img(grouped_array,0,0)
            cropped_array=self.cp.apply_group_crop(grouped_array,roi_crop)
            cropped_mask=self.cp.apply_single_crop(bin_mask,roi_crop)
            cropped_masked_array=self.gs.apply_mask(cropped_array,cropped_mask)
            error_depth_array, error_mask=self.gs.apply_depth_check(cropped_masked_array,exp_depth)
            eda_m=error_depth_array[error_mask]
            amp_data=cropped_masked_array[:,0,:,:]
            ################################################
            self.amp_h=Main_Histogram(amp_data)
            self.amp_h.histo()
            self.amp_h.hist_bins
            self.err_d=Depth_Error_hist(self.amp_h.hist_bins)
            self.err_d.loop_mask(amp_data,error_depth_array)
            name=str(self.edt.exp_numb)+" - Aten: "+exp_aten+" - Dist: "+str(exp_depth)
            self.err_d.plot(name)
            ###################################################
            #from IPython import embed; embed()
            apa_m=amp_data[error_mask]
            self.result_table_list.append(
                                    (exp_aten, exp_depth,
                                     apa_m.mean(), apa_m.std(),
                                     eda_m.mean(), eda_m.std()
                                     )
                                         )
        #columns=['atenuador','exp_depth','amp_mean','amp_std','error_mean','error_std']
        self.df=pd.DataFrame(self.result_table_list,columns=columns)
        self.df.to_excel(filename_export)

    def read_saved_analysis(self,excel_filename):
        self.df=pd.read_excel(excel_filename, index_col=0)
        self.dfr=Dataframe_results(self.df)
        print("Available Column Names",self.dfr.label_names)

    def plot_setup(self,x_axis_col_name='exp_depth',
                        y_mean_column='amp_mean',
                        y_std_column='amp_std',
                        label='atenuador'):

        self.dfr.set_x_axis(x_axis_col_name)
        self.dfr.set_y_axis(y_mean_column,y_std_column)
        self.dfr.set_label(label)

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
        
    def crop_otsu_img(self,grouped_array,frame_num,feature_num):
        global img, refPt, cropping
        refPt=[]
        roi_crop=[]
        cropping = False
        frame_img=grouped_array[frame_num][feature_num]
        ##########################################################
        img=filter.norm(self._check_exploding(frame_img))
        mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        mask2 = (mask==255).reshape((240,320))
        img=img*mask2
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        ##########################################################
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
            crop_label=self.add_label(std_label=True)
        cv2.destroyAllWindows()
        roi_crop.append((roi_coord,crop_label))

        return roi_crop,mask2
    
    def crop_norm_img(self,grouped_array,frame_num,feature_num):
        global img, refPt, cropping
        refPt=[]
        roi_crop=[]
        cropping = False
        frame_img=grouped_array[frame_num][feature_num]
        img=filter.norm(self._check_exploding(frame_img))
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
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
            crop_label=self.add_label(std_label=True)
        cv2.destroyAllWindows()
        roi_crop.append((roi_coord,crop_label))

        return roi_crop

    def _crop_coord_detect(self,refPt,img):
        x_max=max(refPt[0][0],refPt[1][0])
        x_min=min(refPt[0][0],refPt[1][0])
        y_max=max(refPt[0][1],refPt[1][1])
        y_min=min(refPt[0][1],refPt[1][1])
        roi = img[y_min:y_max, x_min:x_max]
        roi_coord = [x_min,y_min,x_max,y_max]
        return roi, roi_coord

    def add_label(self,std_label=True):
        if std_label==False:
            label=input("Set Label Name for cropped region:")
        else:
            label="std_label"
        return label

    def apply_group_crop(self,grouped_array,roi_crop):
        if len(roi_crop)==1:
            roi_coord=roi_crop[0][0]
            # [x_min,y_min,x_max,y_max]
            group_crop=grouped_array[:,:,roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
        elif len(roi_crop)<1:
            print("Error - Crop cannot be performed without ROI coordinates")
            print("roi_crop input value = ",roi_crop)
            group_crop="error"
        return group_crop

    def apply_single_crop(self,simple_array,roi_crop):
        if len(roi_crop)==1:
            roi_coord=roi_crop[0][0]
            # [x_min,y_min,x_max,y_max]
            simple_crop=simple_array[roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
        elif len(roi_crop)<1:
            print("Error - Crop cannot be performed without ROI coordinates")
            print("roi_crop input value = ",roi_crop)
            simple_crop="error"
        return simple_crop

class Feature_show():

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
    
    def apply_depth_check(self,any_grouped_array,exp_depth,plot_show=False):

        error_depth_array=any_grouped_array[:,2,:,:]-exp_depth/1000
        eda=error_depth_array
        print(eda.shape[0]*eda.shape[1]*eda.shape[2])
        print(exp_depth/1000)
        eda=error_depth_array.copy()
        eda_mask=abs(eda[:,:,:])<0.08
        print(eda_mask.sum())
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
                                linestyle=ls, label=label_name)
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

class Amplitude_hist(Main_Histogram):
    
    def __init__(self,data):
        self.hist_min=0
        if data.max()>1000:
            self.hist_max=data.max()
        else:
            self.hist_max=1000

class Depth_Error_hist(Main_Histogram):
    def __init__(self,bins):
        self.bin_pair=[]
        for i in range(len(bins)):
            if i+1<len(bins):
                self.bin_pair.append([bins[i],bins[i+1]])

    def bins_mask(self,amp_data,bin_min,bin_max):
        lower_mask=amp_data[:,:,:]>=bin_min
        high_mask=amp_data[:,:,:]<=bin_max
        final_mask=np.logical_and(lower_mask,high_mask)
        return final_mask
    
    def apply_mask(self,final_mask,error_data):
        mask_data=error_data[final_mask]
        if len(mask_data.shape)==1:
            mask_count=len(mask_data)
        elif len(mask_data.shape)==2:
            mask_count=mask_data.shape[0]*mask_data.shape[1]
        elif len(mask_data.shape)==3:
            mask_count=mask_data.shape[0]*mask_data.shape[1]*mask_data.shape[2]
        mask_mean=mask_data.mean()
        mask_std=mask_data.std()
        return mask_mean,mask_std,mask_count

    def loop_mask(self,amp_data,error_data):
        self.final_count=[]
        for pair in self.bin_pair:
            final_mask=self.bins_mask(amp_data,pair[0],pair[1])
            mask_mean,mask_std,mask_count=self.apply_mask(final_mask,error_data)
            if mask_count!=0: 
                self.final_count.append([pair[1],mask_count,mask_mean,mask_std])
    
    def plot(self,name):
        fct=self.final_count
        fig, ax = plt.subplots(figsize=(7, 4))
        ls = 'dotted'
        ax.set_title("Amplitude vs Depth_Error Mean+-Std"+name)
        #ax.text(0,0,name, va="top", ha="left")
        for i in range(len(fct)):
            ax.errorbar(fct[i][0],
                        fct[i][2],
                        yerr=fct[i][3],marker='o',
                        linestyle=ls,label=fct[i][0]
                        )
            ax.legend(loc='lower right')
            #+"-Exp: "+name
            
            
        plt.show()
        
if __name__ == "__main__":
    main()
    pass