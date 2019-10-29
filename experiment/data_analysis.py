import sys
### Modify sys path if the tof project folder is not in PATH
#print(sys.path)
sys.path.append("D:\\tof")

import numpy as np
import cv2
import pandas as pd
#### Folder and data libraries
import os
#### Plot Libraries
import matplotlib.pyplot as plt

#### Customized Libraries - Luhm
from bin_opener.input_data import Input_data, Mfolder
from experiment.exp_data import Exp_data
from experiment.histogram import Main_Histogram, Depth_Error_hist
from viewer.show_n_crop import Feature_show, Crop

#####
"""
Objectives:
-- Add Multiple Crop Feature
-- Modify Output file crop multiple crop names for a single experiment.
-- Adjust Errormap to group results by label and experiments 
"""
def main():
    ### File path to load data of all experiment setup data loading
    file_path="D:/tof/Experimento_TOF_Texas.xlsx"
    ### folder path to load data of a single TOF experiment 
    #folder_path='D:/Codes/Videos_tof/Exprimentos_atenuadores/Exp004'
    ### File path to load data of all TOF exesetup datperiment a loading
    all_folder_path='D:/Codes/Videos_tof/Experimentos_atenuadores/'
    da=Data_analysis(all_folder_path,file_path)
    
    #################################################################
    ### Enable Crop
    #################################################################
    #crop_activate_list=[0,3,8,11,16,19] Single Object Files
    #crop_activate_list=[1,5,9] # Multiple Perpendicular Objects
    #crop_activate_list=[0,1,4,5,8,9,12,15,18,21]
    #da.select_crop_folders(crop_activate_list,crop_object_num=4)
    output="d:/tof/tof_exp_multi_perp.xlsx"
    #da.save_crop_coord(excel_filename=output)

    #################################################################
    ### Load Crop Object
    #################################################################
    da.load_crop_coord(excel_filename=output, source="excel")

    #################################################################
    ### Divive crop into patches and export to excel
    #################################################################
    pixels=1
    da.crop_into_patches(pixels=pixels)
    output2="d:/tof/coord_patches_pixel_num_"+str(pixels)+".xlsx"
    #da.df_patch_coord.to_excel(output2)
    #output2="d:/tof/coord_patches_pixel_num_5.xlsx"
    #################################################################
    ### Load patch coord
    #################################################################
    da.load_crop_patches(excel_filename=output2,source="dataframe")
    crop_list=da.patches_list()
    #from IPython import embed;embed()
    #################################################################
    ### Run analysis
    #################################################################
    da.run_data_analysis(folders_numb=12,
                         frame_limit=30,
                         crop_label_list=crop_list,
                         mode="patch")
    #################################################################
    ### Plot analysis
    #################################################################
    df=pd.DataFrame(da.final_count_pack,columns=["crop_label","aten","err_depth_mean","err_depth_std","amp_data_mean","amp_data_avg"])
    df.to_excel("d:/tof/coord_patches_mean_std_"+str(pixels)+".xlsx")
    #Data_analysis.filter_plot(da.final_count_pack,da.depth_list,da.aten_list)
    #from IPython import embed;embed()

    #################################################################
    ### Load Crop
    #################################################################
    # depth_list=["1013","3009","4834"]
    # aten_list=["Sem filtro","Pelicula 1","Pelicula 2",
    #            "Pelicula 3","Pelicula 4","Pelicula 5"]
    #aten_list=["Sem filtro","Fume 1","Fume 2",
    #           "Pelicula 1","Pelicula 2","Pelicula 3","Pelicula 4","Pelicula 5"]
    
    # for depth in depth_list:
    #     lst=[depth]
    #     for aten in aten_list:
    #         lst2=[aten]
    #         aten_pack=Data_analysis.filter_by_atenuator(self.final_count_pack,aten_list=lst2)
    #         depth_pack=Data_analysis.filter_by_depth(aten_pack,depth_list=lst)
    #         Data_analysis.error_plot(depth_pack)
   
    #columns=['atenuador','exp_depth','amp_mean','amp_std','error_mean','error_std'],
    #filename_export="d:/tof/output.xlsx"

class Data_analysis():
    def __init__(self,all_exp_folder,exp_setup_filepath,frame_limit=30):
        self.mfld=Mfolder(all_exp_folder)
        self.mfld.swipe_folders()
        self.edt=Exp_data(exp_setup_filepath)
        self.resumed_exp_data=[]
        self.detailed_exp_data=[]
        self.roi_crop_pack=[]
        self.final_count_pack=[]
        self.frame_limit=frame_limit
        self.fs=Feature_show()
        self.cp=Crop()

    def single_folder(self,folder_count):
        folder_path=self.mfld.folder_path_list[folder_count]
        self.edt.get_exp_numb(folder_path)
        self.exp_depth=self.edt.get_exp_data(self.edt.exp_numb,"dist_mm")
        self.exp_aten=self.edt.get_exp_data(self.edt.exp_numb,"Atenuador")
        self.exp_ang=self.edt.get_exp_data(self.edt.exp_numb,"angulo")
        self.idt=Input_data(folder_path)
        self.grouped_array=self.idt.reshaped_grouped()[:self.frame_limit,:,:,]

    def select_crop_folders(self,crop_folder_list,crop_mode="manual",crop_object_num=1):
        if (crop_mode=="auto" or crop_mode=="manual")==False:
            print("Available Crop modes are: auto or manual - Switching to auto mode")
            crop_mode="auto"
        else:
            self.crop_activate_list=crop_folder_list
            for crop_fold_num in crop_folder_list:
                self.single_folder(crop_fold_num)
                img, mask = Feature_show().dual_otsu(self.grouped_array[0][0],single_data=True)
                if crop_mode=="auto":
                    self.auto_crop( frame_num=0,feature_num=0)
                elif crop_mode=="manual":
                    for crop_obj in range(crop_object_num): 
                        self.manual_crop( frame_num=0,feature_num=0,
                                            std_label=True,crop_obj=crop_obj)
            self.save_crop_coord()
        
    def manual_crop(self, frame_num=0,feature_num=0,
                      std_label=False,crop_obj=0): 
        #frame_num=0 #Uses the first frame
        #feature_num=0 where 0 = Amplitude      
        if crop_obj==0:
            self.img_o, self.bin_mask=self.fs.dual_otsu(self.grouped_array,frame_num=0,feature_num=0,single_data=False)
            #self.img_o, self.bin_mask=self.fs.dual_otsu(self.grouped_array,frame_num,feature_num,crop_enhance=True)
        self.roi_crop, self.img_o=self.cp.crop_img(
                                        self.img_o,std_label, self.exp_depth, 
                                        self.exp_aten,self.exp_ang, crop_obj=crop_obj
                                        )
            
        self.roi_crop_pack.append(self.roi_crop)

    def auto_crop(self, frame_num=0,feature_num=0,std_label=True):
            self.img_o, self.bin_mask=self.fs.dual_otsu(self.grouped_array,frame_num=0,feature_num=0,single_data=False)
            if std_label:
                bbox_rename=False
            else:
                bbox_rename=True
            bndboxes=self.cp.get_bbox(self.img_o,pixel_area=10)
            _, bbox_names=self.cp.show_bbox(self.img_o,bndboxes,bbox_rename=bbox_rename)

            for num,bbox_name in enumerate(bndboxes):
                self.roi_crop.append(list(bndboxes),bbox_name,self.exp_aten,self.exp_depth,self.exp)
                self.roi_crop_pack.append(self.roi_crop)
        
    def save_crop_coord(self,excel_filename="D:/tof/outputs/roi_coord.xlsx"):
        array=np.array(self.roi_crop_pack)
        array_p1=array[:,:,0][:,0]
        array_p2=array[:,:,1:][:,0]
        df1=pd.DataFrame(array_p1,columns=["roi_coord"])
        df2=pd.DataFrame(array_p2,columns=["label","aten","depth","ang"])
        self.df_coord=pd.concat([df1,df2],axis=1)
        self.df_coord.to_excel(excel_filename)

    def load_crop_coord(self, excel_filename="D:/tof/outputs/roi_coord.xlsx",
                       dataframe=pd.DataFrame(),
                       source="dataframe"):
        #source ="dataframe" / "excel"
        if source =="dataframe":
            if len(dataframe)==0:
                self.df_coord=self.df_coord
            else:
                self.df_coord=dataframe
        elif source =="excel":
            self.df_coord=pd.read_excel(excel_filename, index_col=0)
            a=np.array(self.df_coord['roi_coord'])
            for row in range(a.shape[0]):
                row_ls=a[row].split("[")[1].split("]")[0].split(",")
                for num,row_elem in enumerate(row_ls):
                    row_ls[num]=int(row_elem)
                row_ls=np.array(row_ls)
                a[row]=row_ls
            self.df_coord['roi_coord']=a

    def load_crop_patches(self, excel_filename="",
                       dataframe=pd.DataFrame(),
                       source="dataframe"):
        if source =="dataframe":
            if len(dataframe)==0:
                self.df_patch_coord=self.df_patch_coord
            else:
                self.df_patch_coord=dataframe
        elif source =="excel":
            self.df_patch_coord=pd.read_excel(excel_filename, index_col=0)
            a=np.array(self.df_patch_coord['patch_coord'])
            for row in range(a.shape[0]):
                row_ls=a[row].split("[")[1].split("]")[0].split(",")
                for num,row_elem in enumerate(row_ls):
                    row_ls[num]=int(row_elem)
                row_ls=np.array(row_ls)
                a[row]=row_ls
            self.df_patch_coord['patch_coord']=a
        
    def crop_into_patches(self,pixels=9):
        self.pixel=pixels
        patch_coord_pack=[]
        for num,bndbox in enumerate(self.df_coord['roi_coord']):
            count=0
            for y in range((bndbox[3]-bndbox[1])//pixels):
                y_0 = (y)*pixels + bndbox[1]
                y_1 = (y+1)*pixels + bndbox[1]
                pixels_error = []
                for x in range((bndbox[2]-bndbox[0])//pixels):
                    x_0 = (x)*pixels + bndbox[0]
                    x_1 = (x+1)*pixels + bndbox[0]
                    patch_coord=[x_0,y_0,x_1,y_1]

                    patch_coord_pack.append(
                        [self.df_coord['roi_coord'][num],
                         self.df_coord['label'][num],
                         self.df_coord['aten'][num],
                         self.df_coord['depth'][num],
                         self.df_coord['ang'][num],
                         count, pixels,patch_coord ]
                                      )
                    count=count+1
        self.df_patch_coord=pd.DataFrame(patch_coord_pack,
                                         columns=['roi_coord_obj',
                                                  'label_obj',
                                                  'aten',
                                                  'depth',
                                                  'ang',
                                                  'patch_num','patch_pixel','patch_coord'])

    def patches_list(self):

        self.df_patch_coord['full_label']='None'
        ls=[]
        ls1=[]
        for row in range(self.df_patch_coord.shape[0]):
            valA=self.df_patch_coord.loc[row,'label_obj']
            valB=self.df_patch_coord.loc[row,'patch_num']
            valC=self.df_patch_coord.loc[row,'depth']

            strA=str(valA); strB=str(valB); strC=str(valC)
            srtD=strA+"_patch_"+strB+"_depth="+strC
            self.df_patch_coord.loc[row,'full_label']=srtD
            if srtD not in ls1:
                ls.append([valA,valB,valC])
                ls1.append(srtD)
        return ls1

    def get_crop_coord(self,crop_label="std_label-0",aten_num=1):
        if aten_num==0:
            aten_lb="Sem Filtro"
        elif aten_num==1:
            aten_lb="Pelicula 1"
        depth=self.df_coord['depth']==self.exp_depth
        label=self.df_coord['label']==crop_label
        aten=self.df_coord['aten']==aten_lb
        and_check=self.df_coord[((depth & label)& aten)]
        if len(and_check)==0:
            print("Crop coordinates not found")
        else:
            roi_coord=str(and_check['roi_coord'])

            roi_coord=roi_coord.split("[")[1].split("]")[0].split(",")
            for num,coord in enumerate(roi_coord):
                roi_coord[num]=int(coord)
        return roi_coord

    def get_patch_coord(self,crop_label="std_label-0",depth="",aten_num=1):
        if aten_num==0:
            aten_lb="Sem Filtro"
        elif aten_num==1:
            aten_lb="Pelicula 1"
        
        label=self.df_patch_coord['full_label']==crop_label
        aten=self.df_patch_coord['aten']==aten_lb
        #dep=self.df_patch_coord['depth']==depth
        and_check=self.df_patch_coord[(label & aten)]

        if int(and_check['depth'])!=depth:
            roi_coord=0
        else:
            if len(and_check)==0:
                print("Crop coordinates not found")
            else:
                roi_coord=str(and_check['patch_coord'])
                roi_coord=roi_coord.split("[")[1].split("]")[0].split(",")
                for num,coord in enumerate(roi_coord):
                    roi_coord[num]=int(coord)
        return roi_coord

    def run_data_analysis(self,
                          folders_numb=15,
                          frame_limit=30,
                          crop_label_list=["std_label-0"],mode="object"):
        #### Available crop_mode = ["single","multiple"]
        self.aten_list=[]
        self.depth_list=[]
        for folder_count in range(folders_numb):
            self.single_folder(folder_count)
            ################
            amp_img, self.bin_mask=self.fs.dual_otsu(self.grouped_array[0][0],single_data=True)
            amp_img=((amp_img/amp_img.max())*255).astype("uint8")
            amp_img=cv2.cvtColor(amp_img,cv2.COLOR_GRAY2BGR)
            img_name=("ExpA-Aten: "+str(self.exp_aten)+" Depth: "+str(self.exp_depth))
            cv2.namedWindow(img_name,cv2.WINDOW_KEEPRATIO)
            pack1=[]
            ##############
            #self.fs.filters_eval(self.grouped_array,self.exp_depth,self.exp_aten)

            if self.exp_aten not in self.aten_list:
                self.aten_list.append(self.exp_aten)

            if self.exp_depth not in self.depth_list:
                self.depth_list.append(self.exp_depth)

            for crop_label in crop_label_list:
                #from IPython import embed;embed()
                #print(self.df_patch_coord)
                if self.exp_aten=="Sem Filtro":
                    aten_num=0
                else:
                    aten_num=1
                    
                if mode=="object":
                    roi_coord=self.get_crop_coord(crop_label=crop_label,aten_num=aten_num)


                elif mode=="patch":
                    roi_coord=self.get_patch_coord(crop_label=crop_label,depth=self.exp_depth,aten_num=aten_num)
                
                if roi_coord!=0:
            
                    error_depth_array,amp_data=self.single_object_analysis(roi_coord,crop_label)
                    pack=[crop_label,self.exp_aten,error_depth_array.mean(), error_depth_array.std(),
                                                    amp_data.mean(), amp_data.std()]
                    pt1=(roi_coord[0],roi_coord[1])
                    pt2=(roi_coord[2],roi_coord[3])
                    pt3=(roi_coord[2]-roi_coord[0],roi_coord[3]-roi_coord[1])
                    if abs(error_depth_array.mean())>0.2:
                        clr=(0,0,255)
                        text=""
                    elif (abs(error_depth_array.mean())<=0.2 and abs(error_depth_array.mean())>0.1):
                        clr=(80,127,255)
                        text=""

                    elif (abs(error_depth_array.mean())<=0.1 and abs(error_depth_array.mean())>0.05):
                        clr=(0,215,255)
                        text=""
                    else:
                        clr=(0,255,0)
                        text=""

                    self.final_count_pack.append(pack)
                    cv2.rectangle(amp_img,pt1=pt1,pt2=pt2,color=clr,thickness=1)
                    cv2.putText(amp_img,text,pt3,cv2.FONT_HERSHEY_SIMPLEX, 1.0,color=clr)
            amp_img=cv2.resize(amp_img,dsize=None,fx=5,fy=5)
            cv2.imshow(img_name,amp_img)
            filename=("Exp-Aten_"+str(self.exp_aten)+"_Depth_"+str(self.exp_depth)+"_PixelPatch_"+str(self.pixel))
            filename="D:/tof/outputs/img"+filename+".jpg"
            cv2.imwrite(filename,amp_img)
            cv2.waitKey()
        cv2.destroyAllWindows()
        
                #self.resumed_exp_data.append(( self.exp_aten, self.exp_depth,
                 #                               self.exp_ang,crop_label,
                  #                              error_depth_array.mean(), error_depth_array.std(),
                   #                             amp_data.mean(), amp_data.std()))
                

                #final_count,name=self._apply_histogram(amp_data,error_depth_array)
                                               
                #final count list format [amplitude,mask_count,mask_mean,mask_std]

    def single_object_analysis(self,roi_coord,crop_label):

        cropped_array=self.cp.multi_feature_crop(self.grouped_array,roi_coord)
        cropped_mask=self.cp.single_feature_crop(self.bin_mask,roi_coord)
        if cropped_mask.sum()==0:
            cropped_masked_array=cropped_array
        else:   
            cropped_masked_array=self.fs.apply_mask(cropped_array,cropped_mask)

        error_depth_array=self.fs.apply_depth_check(cropped_masked_array, self.exp_depth)

        #### Check Effects of error mask over depth array and amp_data
        amp_data=cropped_masked_array[:,0,:,:]

        return error_depth_array,amp_data

    def _apply_histogram(self,amp_data,error_depth_array,plot=False):
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
        name=(e_name+str(self.edt.exp_numb)+
              " - Aten: " + self.exp_aten+
              " - Depth: " + str(self.exp_depth)+
              " - Ang: "  + self.exp_ang)
        if plot==True:
            self.err_d.plot(name,final_count)
        return final_count,name

    def read_saved_analysis(self,excel_filename):
        self.df=pd.read_excel(excel_filename, index_col=0)
        self.dfr=Dataframe_results(self.df)
        print("Available Column Names",self.dfr.label_names)
    
    @staticmethod
    def filter_plot(final_count_pack,depth_list,aten_list):
        from IPython import embed;embed()
        for depth in depth_list:
            lst=[str(depth)]
            for aten in aten_list:
                lst2=[aten]
                aten_pack=Data_analysis.filter_by_atenuator(final_count_pack,aten_list=lst2)
                depth_pack=Data_analysis.filter_by_depth(aten_pack,depth_list=lst)
                #from IPython import embed; embed()
                Data_analysis.error_plot(depth_pack)

    @staticmethod
    def error_plot(count_pack):
        jitter=2
        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Amplitude vs Depth_Error Mean+-Std")         
        ls = 'dotted'
        for exp_num,experiment in enumerate(count_pack):

            data=np.array(experiment[0])
            if len(data)!=0:
                exp_name=experiment[1]
                label=experiment[2]
                label_name=exp_name+"- Label: "+label
                x_axis_values=data[:,0]+exp_num*jitter
                y_mean=data[:,2]
                y_std=data[:,3]

                ax.errorbar(x_axis_values,
                                        y_mean,
                                    yerr=y_std,
                                    linestyle=ls,
                                    marker="o",
                                    label=label_name)
  
                ax.legend(loc='lower right')
                f_name=exp_name.replace(" - ","_")
                f_name=f_name.replace(": ","_")



                #from IPython import embed; embed()
                file_name="C:/Users/ricar/Desktop/Output/"+f_name+"_crop4.jpg"
                plt.savefig(file_name)
        #ax.set_xlim((0, 1000))
        #ax.set_ylim((-0.200, 0.200))
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
                depth=exp_name.split("Depth: ")[1]
                depth=depth.split(" - ")[0]
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



        
if __name__ == "__main__":
    main()
    pass