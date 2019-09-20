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
"""Objectives:
Select experiment folder - OK
Select file with experiment setup data - OK
Related experiment folder within experiment setup data - OK
Upload bin and convert into array - OK
Manually limit desired frames to analyse - OK
Manually select desired features - OK
-- Get Amplitude Values from each experiment - OK
-- Apply otsu threshold in amplitude values as mask - Pending
Save average feature values and std dev for each experiment and reference object detected by the
 initial mask. - OK
Create a bounding box over reference objects which did not appeared in the first mask
Save average feature values and std dev for each experiment of reference objects in the bounding box
Calculate the depth error measurement based on the laser measured
Plot avg and std feature: 
-- per experiment distance, filter type and reference object
-- c
Plot
"""
def main():
    ### File path for experiment setup data loading
    file_path="C:/Users/ricar/Downloads/Experimento_TOF_Texas.xlsx"
    ### folder path for experiment TOF data loading
    folder_path='D:/Codes/Videos_tof/Experimentos_atenuadores/Exp004'

    all_folder_path='D:/Codes/Videos_tof/Experimentos_atenuadores/'

    mfld=Mfolder(all_folder_path)
    mfld.swipe_folders()
    edt=Exp_data(file_path)
    
    crop_activate_list=[0,5,10]
    result_table_list=[]
    for folder_count in range(15):
        folder_path=mfld.folder_path_list[folder_count]
        edt.get_exp_numb(folder_path)
        exp_depth=edt.get_exp_data(edt.exp_numb,"dist_mm")
        exp_aten=edt.get_exp_data(edt.exp_numb,"Atenuador")
        idt=Input_data(folder_path)
        frame_limit=30
        grouped_array=idt.reshaped_grouped()[:frame_limit,:,:,]
        gs=Group_show()
        if folder_count in crop_activate_list:
            roi_crop=gs.crop_img_frame(grouped_array,0,0)
        cropped_array=gs.apply_group_crop(grouped_array,roi_crop)
        error_depth_array=gs.apply_depth_check(cropped_array,exp_depth)
        eda=error_depth_array
        amp_data=cropped_array[:,0,:,:]
        result_table_list.append((exp_aten,exp_depth,amp_data.mean(),amp_data.std(),eda.mean(),eda.std()))
    
    columns=['atenuador','exp_depth','amp_mean','amp_std','error_mean','error_std']
    df=pd.DataFrame(result_table_list,columns=columns)
    df.to_excel("d:/tof/output2.xlsx")
    #file_path="d:/tof/output2.xlsx"
    #df=pd.read_excel(file_path, index_col=0)
    
    sdf=Specific_DF(df)
    sdf.set_x_axis('exp_depth')
    sdf.set_y_axis('amp_mean','amp_std')
    sdf.set_label('atenuador')
    #sdf.plot_all()
    #print("Available Names",sdf.label_names)
    sdf.plot_single()
    
    
        #result_table_list=[atenuador,depth_experimento,amp_mean,amp_std,error_mean,error_std]

    #from IPython import embed; embed()

    #gs.feature_show(cropped_array)
    #folder_path='D:/Codes/Videos_tof/Experiments_Vel/Exp_0_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512'

class Group_show():

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
            img[mask]=exp_threshold+img.std()
        return img

    def _click_and_crop(self,event, x, y, flags, param):
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

    def _crop_coord_detect(self,refPt,img):
        x_max=max(refPt[0][0],refPt[1][0])
        x_min=min(refPt[0][0],refPt[1][0])
        y_max=max(refPt[0][1],refPt[1][1])
        y_min=min(refPt[0][1],refPt[1][1])
        roi = img[y_min:y_max, x_min:x_max]

        roi_coord = [x_min,y_min,x_max,y_max]
        return roi, roi_coord

    def feature_show(self,grouped_array):
        for i in range(grouped_array.shape[0]):
            self.std_window_show("Amplitude",filter.norm(self._check_exploding(grouped_array[i][0])))
            self.std_window_show("Ambient",filter.norm(self._check_exploding(grouped_array[i][1])))
            self.std_window_show("Depth",filter.norm(self._check_exploding(grouped_array[i][2])))
            #grouped_array[i][2]))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def add_label(self,std_label=True):
        if std_label==False:
            label=input("Set Label Name for cropped region:")
        else:
            label="std_label"
        return label

    def crop_img_frame(self,grouped_array,frame_num,feature_num):
        global img, refPt, cropping
        refPt=[]
        roi_crop=[]
        cropping = False
        frame_img=grouped_array[frame_num][feature_num]
        ##########################################################
        img=filter.norm(self._check_exploding(frame_img))
        print("img in ", img.shape)
        mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        print("mask ", mask.shape)
        mask2 = (mask==255).reshape((240,320))
        print("mask2 ", mask2.shape)
        img=img*mask2
        #img=img[mask2].reshape((240,320))
        print("img out ", img.shape)
        #from IPython import embed;embed()
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        ##########################################################
        img_clone = img.copy()
        cv2.namedWindow('Image',cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('Image', self._click_and_crop)

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
        
    def apply_group_crop(self,grouped_array,roi_crop):
        if len(roi_crop)==1:
            roi_coord=roi_crop[0][0]
            # [x_min,y_min,x_max,y_max]
            group_crop=grouped_array[:,:,roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
        return group_crop

    def apply_depth_check(self,any_grouped_array,exp_depth,plot_show=False):
        error_depth_array=any_grouped_array[:,2,:,:]-exp_depth/1000
        eda=error_depth_array.copy()
        
        if plot_show==True:
            for i in range(eda.shape[0]):
                ###### Code under development - 
                img=eda[i]
                img=filter.norm(img)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                self.std_window_show("Depth check",img)
                cv2.waitKey()
                cv2.destroyWindow("Depth check")
            
        return error_depth_array


class Specific_DF():
    ##
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
            

if __name__ == "__main__":
    main()
    pass