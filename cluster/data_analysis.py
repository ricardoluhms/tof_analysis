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
from filters.utils import filter
from filters.utils import edge_filter as edf
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
    
    #################################################################
    ### Enable Crop
    #################################################################
    #crop_activate_list=[0,3,8,11,16,19] Single Object Files
    #crop_activate_list=[0,1,4,5,8,9] # Multiple Perpendicular Objects
    #crop_activate_list=[0,1,4,5,8,9,12,15,18,21]
    #da.select_crop_folders(crop_activate_list,crop_object_num=4)
    output="d:/tof/tof_exp_multi_perp.xlsx"
    #da.save_crop_coord(excel_filename=output)

    #################################################################
    ### Load Crop
    #################################################################
    da.load_crop_coord(excel_filename=output, source="excel")
    #################################################################
    ### Run analysis
    #################################################################
    da.run_data_analysis(folders_numb=12,frame_limit=30,
                         crop_label_list=["fita_3M","mercosul","reflet","disco"])

    #################################################################
    ### Plot analysis
    #################################################################
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

    def select_crop_folders(self,crop_folder_list,crop_object_num=1):
        self.crop_activate_list=crop_folder_list
        for crop_fold_num in crop_folder_list:
            self.single_folder(crop_fold_num)
            for crop_obj in range(crop_object_num): 
                self.activate_crop( frame_num=0,feature_num=0,
                                    std_label=True,crop_obj=crop_obj)
        self.save_crop_coord()

    def activate_crop(self, frame_num=0,feature_num=0,
                      std_label=False,crop_obj=0): 
        #frame_num=0 #Uses the first frame
        #feature_num=0 where 0 = Amplitude      
        if crop_obj==0:
            self.img_o, self.bin_mask=self.fs.get_otsu(self.grouped_array,frame_num,feature_num)   
        self.roi_crop=self.cp.crop_img(
            self.img_o,std_label, self.exp_depth, 
            self.exp_aten,self.exp_ang, crop_obj=crop_obj
            )
        self.roi_crop_pack.append(self.roi_crop)
        
    def save_crop_coord(self,excel_filename="D:/tof/outputs/roi_coord.xlsx"):
        array=np.array(self.roi_crop_pack)
        array_p1=array[:,:,0][:,0]
        array_p2=array[:,:,1:][:,0]
        df1=pd.DataFrame(array_p1,columns=["roi_coord"])
        df2=pd.DataFrame(array_p2,columns=["label","aten","depth","ang"])
        self.df_coord=pd.concat([df1,df2],axis=1)
        self.df_coord.to_excel(excel_filename)

    def load_crop_coord(self,
                       excel_filename="D:/tof/outputs/roi_coord.xlsx",
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

    def run_data_analysis(self,
                          folders_numb=15,
                          frame_limit=30,
                          crop_label_list=["std_label-0"]):
        #### Available crop_mode = ["single","multiple"]
        self.aten_list=[]
        self.depth_list=[]
        for folder_count in range(folders_numb):
            self.single_folder(folder_count)
            _, self.bin_mask=self.fs.get_otsu(self.grouped_array,0,0)
            self.fs.filters_eval(self.grouped_array,self.exp_depth,self.exp_aten)
            ##############
            if self.exp_aten not in self.aten_list:
                self.aten_list.append(self.exp_aten)

            if self.exp_depth not in self.depth_list:
                self.depth_list.append(self.exp_depth)
            ##############

            for crop_label in crop_label_list:
                if self.exp_aten=="Sem Filtro":
                    aten_num=0
                else:
                    aten_num=1
                roi_coord=self.get_crop_coord(crop_label=crop_label,aten_num=aten_num)
                error_depth_array,eda_m,amp_data,apa_m,status=self.single_object_analysis(
                    roi_coord,crop_label)
                #  from IPython import embed; embed()
                if status:
                    self.resumed_exp_data.append(( self.exp_aten, self.exp_depth,
                                                self.exp_ang,
                                                apa_m.mean(), apa_m.std(),
                                                eda_m.mean(), eda_m.std()))
                else:
                    print("Threshold Error- Aten= ", self.exp_aten,
                          " - Depth= ", self.exp_depth,
                          " - Ang= ", self.exp_ang,
                          " - Label= ", crop_label)




                final_count,name=self._apply_histogram(amp_data,error_depth_array)
                
                self.final_count_pack.append((final_count,name,crop_label))
                #final count list format [amplitude,mask_count,mask_mean,mask_std]

    def single_object_analysis(self,roi_coord,crop_label):

        cropped_array=self.cp.multi_feature_crop(self.grouped_array,roi_coord)
        cropped_mask=self.cp.single_feature_crop(self.bin_mask,roi_coord)

        if cropped_mask.sum()==0:
            cropped_masked_array=cropped_array
        else:   
            cropped_masked_array=self.fs.apply_mask(cropped_array,cropped_mask)
        error_depth_array, error_mask=self.fs.apply_depth_check(
                        cropped_masked_array,
                        self.exp_depth,
                        self.exp_aten,
                        crop_label,
                        plot_show=False)
        #### Check Effects of error mask over depth array and amp_data
        amp_data=cropped_masked_array[:,0,:,:]
        if error_mask.sum()==0:
            apa_m=np.array([0])
            eda_m=np.array([0])
            status=False
        else:
            apa_m=amp_data[error_mask]
            eda_m=error_depth_array[error_mask]
            status=True
        return error_depth_array,eda_m,amp_data,apa_m,status

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

class Feature_show():
    def __init__(self):
        pass

    @staticmethod
    def std_window_show(window_name,array):
        cv2.namedWindow(window_name,cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(window_name,array)
    
    @staticmethod
    def single_window(window_name,array):
        cv2.namedWindow(window_name,cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(window_name,array)
        cv2.waitKey()
        cv2.destroyWindow(window_name)

    @staticmethod
    def _check_exploding(img):
        exp_threshold=img.mean()+3*img.std()
        mask=img[:,:]>exp_threshold
        sum_mask=mask.sum()
        if sum_mask>1:
            img[mask]=exp_threshold#+img.std()
        return img
    @staticmethod
    
    def conv_weight(img,kmin=0.1,kmax=1):
        kser1=np.array([kmax,kmax,kmax])
        kser2=np.array([kmax,kmin,kmax])
        kernel=np.vstack((kser1,kser2))
        kernel=np.vstack((kernel,kser1))
        img=cv2.filter2D(img,-1,kernel)/kernel.sum()
        img=img/img.max()
        return img
        #from IPython import embed;embed()

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
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        return img, mask

    def get_edge(self,grouped_array,frame_num,feature_num):
        frame_img=grouped_array[frame_num][feature_num]
        shap=frame_img.shape
        edge=edf(shap[0],shap[1])
        ##########################################################
        img=edge.apply(frame_img)
        #from IPython import embed; embed()
        return img#

    def filters_eval(self,any_grouped_array,exp_depth,exp_aten):
        name0=("Depth: "+str(exp_depth)+"-At: "+exp_aten)
        aga=any_grouped_array
        a_shape=aga.shape
        names=["Amp.","Amp.Edge","Amp.Otsu.Mask","Amp.Otsu.Out","Depth","D.Error","Depth.Edge"]
        f_names=[]
        #### Amp Filters
        amp=aga[:,0,:,:]
        otsu_amp_mask=np.zeros(amp.shape)
        otsu_amp_out=np.zeros(amp.shape)
        edge_amp=np.zeros(amp.shape)
        #### Depth Filters
        depth=aga[:,2,:,:]
        depth_error=depth-exp_depth/1000
        edge=edf(a_shape[2],a_shape[3])
        edge_depth=np.zeros(depth.shape)
        for name in names:
            m_name=name+" - "+name0
            f_names.append(m_name)

        for fram in range(a_shape[0]):
            ###Otsu Amp
            ot_img, ot_mask=self.get_otsu(any_grouped_array,fram,0)
            ot_mask=ot_mask.astype("int")*254
            ot_mask=ot_mask/ot_mask.max()
            otsu_amp_mask[fram,:,:]=ot_mask
            otsu_amp_out[fram,:,:]=ot_img
            ###Edge
            edge_depth[fram,:,:]=edge.apply(depth[fram,:,:]).reshape(a_shape[2],a_shape[3])
            edge_amp[fram,:,:]=edge.apply(amp[fram,:,:]).reshape(a_shape[2],a_shape[3])
            ###Error
            d_err_img=depth_error[fram,:,:]
            ### Amp Norm data
            raw_amp_img=amp[fram,:,:]
            
            amp_img_w=self.conv_weight(raw_amp_img)
            amp_img=self._check_exploding(raw_amp_img)
            amp_img2=self._check_exploding(amp_img_w)

            raw_amp_img=raw_amp_img/raw_amp_img.max()
            amp_img=amp_img/amp_img.max()
            amp_img2=amp_img2/amp_img2.max()

            raw_edge=edge.apply(raw_amp_img).reshape(a_shape[2],a_shape[3])
            check_edge=edge.apply(amp_img).reshape(a_shape[2],a_shape[3])
            conv_check_edge=edge.apply(amp_img2).reshape(a_shape[2],a_shape[3])

            self.std_window_show("Raw Amp Img",raw_amp_img)
            self.std_window_show("Check Amp Img",amp_img)
            self.std_window_show("Conv Check Amp Img",amp_img2)
            self.std_window_show("Raw Edge Amp Img",raw_edge)
            self.std_window_show("Check Edge Amp Img",check_edge)
            self.std_window_show("Conv Check Edge Amp Img",conv_check_edge)


            #self.std_window_show(f_names[2],ot_mask)
            #self.std_window_show(f_names[3],ot_img)
            #self.std_window_show(f_names[4],self._check_exploding(depth[fram,:,:]))
            #self.std_window_show(f_names[5],d_err_img)
            #self.std_window_show(f_names[6],edge_depth[fram,:,:])

            key_pressed = cv2.waitKey(300) & 0xff
            if key_pressed in [32, ord('p')]:
                key_pressed = cv2.waitKey(0) & 0xff
        cv2.destroyAllWindows()

    def apply_depth_check(self,any_grouped_array,exp_depth,exp_aten,exp_label,plot_show=False):

        error_depth_array=any_grouped_array[:,2,:,:]-exp_depth/1000
        depth=any_grouped_array[:,2,:,:]
        eda=error_depth_array
        amp_array=any_grouped_array[:,0,:,:]

        eda=error_depth_array.copy()
        eda_mask=abs(eda[:,:,:])<0.2
        name=("GTD: "+str(exp_depth)+" -Lb: "+exp_label+"-At: "+exp_aten)
        if plot_show==True:
            shap=eda[0,:,:].shape
            edge=edf(shap[0],shap[1])
            for i in range(eda.shape[0]):
                #if exp_aten!="Sem filtro":
                if (exp_aten=="Pelicula 10"):
                    img0, ot_mask=self.get_otsu(any_grouped_array,i,0)
                    
                    ot_mask=ot_mask.astype("int")*254
                    ot_mask=cv2.cvtColor(ot_mask,cv2.COLOR_GRAY2RGB)
                    #from IPython import embed; embed()
                    img=eda[i]
                    img=filter.norm(img)
                    img2=depth[i]
                    img2=filter.norm(img2)
                    img3=amp_array[i]
                    img4=edge.apply(img3)
                    img3=filter.norm(img3)
                    img4=img4.reshape(shap[0],shap[1])
                    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                    #name00=("MaskOtsu "+name)
                    name0=("AmpOtsu "+name)
                    name1=("D.Error "+name)
                    name2=("Depth "+name)
                    name3=("Amp. "+name)
                    name4=("Amp.Edge"+name)
                    self.std_window_show(name0,ot_mask)
                    self.std_window_show(name1,img)
                    self.std_window_show(name2,img2)
                    self.std_window_show(name3,img3)
                    self.std_window_show(name4,img4)
                    key_pressed = cv2.waitKey(300) & 0xff
                    if key_pressed in [32, ord('p')]:
                        key_pressed = cv2.waitKey(0) & 0xff
            if (exp_aten=="Pelicula 10"):
                cv2.destroyWindow(name0)
                cv2.destroyWindow(name1)
                cv2.destroyWindow(name2)
                cv2.destroyWindow(name3)
                cv2.destroyWindow(name4)
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
            cv2.rectangle(img, refPt[0], refPt[1], (0,255,255), 1)
            roi, _=self._crop_coord_detect(refPt,img)
            self.std_window_show('Crop',roi)
    
    def crop_img(self,img_o,std_label,exp_depth,exp_aten,exp_ang,crop_obj=1):
        global img, refPt, cropping
        refPt=[]
        roi_crop=[]
        cropping = False
        img = img_o.copy()
        img_clone = img.copy()
        window_name=("Crop object number= "+str(crop_obj)+
                     " - Depth= "+str(exp_depth)+
                     " - Angle= "+str(exp_aten)) 

        cv2.namedWindow(window_name,cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(window_name, self.click_and_crop)
        while True:
            skip_crop=False
            cv2.imshow(window_name, img)
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
        roi_crop.append((roi_coord,crop_label,exp_aten,exp_depth,exp_ang))
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

    def multi_feature_crop(self,grouped_array,roi_coord):
        group_crop=grouped_array[:,:,roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
        return group_crop

    def single_feature_crop(self,simple_array,roi_coord):
        simple_crop=simple_array[roi_coord[1]:roi_coord[3],roi_coord[0]:roi_coord[2]]
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
        self.hist_bins=np.arange(self.hist_min,self.hist_max,15)
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