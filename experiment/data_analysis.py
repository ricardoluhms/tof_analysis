import sys
### Modify sys path if the tof project folder is not in PATH
print(sys.path)
sys.path.append("D:\\tof")
import numpy as np
import cv2
import pandas as pd
#### Folder and data libraries
import os
#### Plot Libraries
import matplotlib.pyplot as plt
from matplotlib import patches as ptc
import matplotlib.cm as cm
#### Customized Libraries - Luhm
from bin_opener.input_data import Input_data, Mfolder
from experiment.exp_data import Exp_data
from experiment.histogram import Main_Histogram, Depth_Error_hist
from viewer.show_n_crop import Feature_show, Crop

def main():
    ### File path to load data of all experiment setup data loading
    #file_path="D:/tof/Experimento_TOF_Texas.xlsx
    #from IPython import embed;embed()   
    file_path="D:/tof/Experimento_TOF_TexasAlcance.xlsx"


    ### folder path to load data of a single TOF experiment 
    #folder_path='D:/Codes/Videos_tof/Exprimentos_atenuadores/Exp004'
    ### File path to load data of all TOF exesetup datperiment a loading
    #all_folder_path='D:/Codes/Videos_tof/Experimentos_atenuadores/

    all_folder_path='D:/Codes/Videos_tof/Experimento_alcance/'
    da=Data_analysis(all_folder_path,file_path)


    ###### Crop list for max reach experiment
    # crop_activate_list=[]
    # for num in range(int(len(da.mfld.folder_path_list)/3)):
    #     if num==0:
    #         a=1
    #     else:
    #         a=a+3
    #     crop_activate_list.append(a)
    
    ######################
    """da.single_folder(1)
    da.histogram3D(da.grouped_array[0][0],da.grouped_array[0][2])"""
    #################################################################
    ### Enable Crop
    #################################################################
    #crop_activate_list=[0,3,8,11,16,19] Single Object Files
    #crop_activate_list=[1,5,9] # Multiple Perpendicular Objects
    #from IPython import embed; embed()
    #crop_activate_list=[0,1,4,5,8,9,12,15,18,21]
    #da.select_crop_folders(crop_activate_list,crop_object_num=5)
    output="d:/tof/tof_exp_multi_alcance_art.xlsx"
    #output="d:/tof/tof_exp_multi_alcance.xlsx"
    #da.save_crop_coord(excel_filename=output)

    #################################################################
    ### Load Crop Object
    #################################################################
    da.load_crop_coord(excel_filename=output, source="excel")
    crop_list=da.df_coord["label"].drop_duplicates().reset_index(drop=True).to_list()
    #################################################################
    ### Divive crop into patches and export to excel
    #################################################################
    #pixels=10
    #da.crop_into_patches(pixels=pixels)
    #output2="d:/tof/coord_patches_pixel_num_"+str(pixels)+".xlsx"
    #da.df_patch_coord.to_excel(output2)
    #output2="d:/tof/coord_patches_pixel_num_5.xlsx"
    #################################################################
    ### Load patch coord
    #################################################################
    #da.load_crop_patches(excel_filename=output2,source="dataframe")
    #crop_list=da.patches_list()
    #from IPython import embed;embed()
    #################################################################
    ### Run analysis
    #################################################################
    #amp_filter_list=[0,3,5,10]
    #amp_filter_list=[0]
    amp_f=3
    da.run_data_analysis(folders_numb=len(da.mfld.folder_path_list),
                        frame_limit=30,
                        crop_label_list=crop_list,
                        mode="object",img_show=False,amp_filt_val=amp_f)
    da.final_count_pack
    df=pd.DataFrame(da.final_count_pack,
                    columns=["crop_label","aten","exp_depth",
                                "err_depth_mean","err_depth_std",
                                "amp_data_mean","amp_data_avg",
                                "depth_data_mean","depth_data_avg",
                                "filter_count","amp_ravel","error_ravel","depth_ravel","digital_f"])


    df.to_excel("d:/tof/outputs2/file_final.xlsx")
    crop_labels=list(df['crop_label'].drop_duplicates().reset_index(drop=True))
    exp_depths=list(df['exp_depth'].drop_duplicates().reset_index(drop=True))
    digitalfitls=list(df['digital_f'].drop_duplicates().reset_index(drop=True))
    atenfilts=list(df['aten'].drop_duplicates().reset_index(drop=True))
    colors = cm.rainbow(np.linspace(0, 1, len(crop_labels)))          

    from IPython import embed;embed()   
    count=24
    for atnum, atenf in enumerate(atenfilts):
        for di_num, dig_f in enumerate(digitalfitls):
            fig=plt.figure()
            #fig, axs = plt.subplots(len(atenfilts), len(digitalfitls))
            fig.set_size_inches(25, 14.18)

            ax1=fig.add_subplot(111,aspect='auto')
            #axs[atnum, di_num].set_title(atenf+" + "+dig_f)
            ax1.set_title("Depth vs Depth Error: "+ atenf+" + "+dig_f, fontsize=22)
            ax1.set_xlabel('Experiment Depth [mm]', fontsize=20)
            ax1.set_ylabel('Depth Error[mm]', fontsize=20)
            ax1.set_ylim(0, 500.0)
            start, end = ax1.get_ylim()
            ax1.yaxis.set_ticks(np.arange(start, end, 25))

            ax1.set_xlim(0, 13000)
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks(np.arange(start, end, 1000))
            ax1.tick_params(axis='both', which='major', labelsize=20)

            
            for num, cplb in enumerate(crop_labels):
                label=df['crop_label']==cplb
                depth=df['exp_depth']!=7501
                filt=df['digital_f']==dig_f
                aten=df['aten']==atenf
                and_check=df[((aten & label)&filt)&depth]
                x=list(np.array(and_check['exp_depth'])+150*num)
                y=list(and_check['err_depth_mean']*1000)
                ym=list(and_check['err_depth_std']*1000)

                ax1.scatter(x,y,c=colors[num])
                ax1.errorbar(x, y, yerr=ym, uplims=False, lolims=True, label=cplb)

            ax1.legend(fontsize=20)
            plt.grid(True)
            fig_name=str(count)+"_Depth_vs_Depth_Error_"+ atenf+"_"+dig_f+".png"
            file_name="D:/tof/outputs2/Art/"+fig_name
            plt.savefig(file_name, dpi=300,format='png')
            count+=1
    #fig.show()

    """pack=[crop_label,self.exp_aten,self.exp_depth,
             round(abs(error_depth_array).mean(),3), round(abs(error_depth_array).std(),3),
             round(amp_data.mean(),1), round(amp_data.std(),1),
             round(depth_data.mean(),3),round(depth_data.std(),3),
             m1.sum(),
             amp_data.ravel(),error_depth_array.ravel(),depth_data.ravel(),
             dif_filt]"""


        #for aten_filt in da.aten_list:
        #    for label_filt in crop_list:
        #        #da.amp_depth_curve(da.final_count_pack,aten_filt,label_filt,mode="Amp",plot_save=True)
        #        da.amp_depth_curve(da.final_count_pack,aten_filt,label_filt,mode="Error",plot_save=True)
    #from IPython import embed;embed()
    #################################################################
    ### Plot analysis
    #################################################################
    #df=pd.DataFrame(da.final_count_pack,columns=["crop_label","aten","err_depth_mean","err_depth_std","amp_data_mean","amp_data_avg"])
    #df.to_excel("d:/tof/coord_patches_mean_std_"+str(pixels)+".xlsx")
    #from IPython import embed;embed()


class Data_analysis():
    def __init__(self,all_exp_folder,exp_setup_filepath,frame_limit=30):
        
        self.mfld=Mfolder(all_exp_folder)
        self.mfld.swipe_folders()
        ### Check Amplitude files and rename
        #self._rename_all_amplitude()
        ####
        self.edt=Exp_data(exp_setup_filepath)
        self.resumed_exp_data=[]
        self.detailed_exp_data=[]
        self.roi_crop_pack=[]
        self.frame_limit=frame_limit
        self.fs=Feature_show()
        self.cp=Crop()

    def _rename_all_amplitude(self):
        for path in self.mfld.folder_path_list:
            self.idt=Input_data(path)
            self.idt.rename_amplitude()
        self.mfld.swipe_folders()
        
    def single_folder(self,folder_count):
        folder_path=self.mfld.folder_path_list[folder_count]
        self.idt=Input_data(folder_path)
        self.edt.get_exp_numb(folder_path)
        self.exp_depth=self.edt.get_exp_data(self.edt.exp_numb,"Depth")
        self.exp_aten=self.edt.get_exp_data(self.edt.exp_numb,"Atenuador")
        self.exp_ang=self.edt.get_exp_data(self.edt.exp_numb,"angulo")
        self.grouped_array=self.idt.reshaped_grouped()[:self.frame_limit,:,:,]

    def select_crop_folders(self,crop_folder_list,crop_mode="manual",crop_object_num=1):
        if (crop_mode=="auto" or crop_mode=="manual")==False:
            print("Available Crop modes are: auto or manual - Switching to auto mode")
            crop_mode="auto"
        else:
            self.crop_activate_list=crop_folder_list
            for crop_fold_num in crop_folder_list:
                self.single_folder(crop_fold_num)
                #from IPython import embed;embed()
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
            aten_lb="No_Optical_Filter"
        elif aten_num==1:
            aten_lb="F1_Optical_Filter"
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
            aten_lb="No_Optical_Filter"
        elif aten_num==1:
            aten_lb="F1_Optical_Filter"
        
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

    def run_data_analysis(self, folders_numb=15, frame_limit=15,
                          crop_label_list=["std_label-0"],mode="object",
                          img_show=False,img_save=False,amp_filt_val=1):
        self.amp_filt_val=amp_filt_val                 
        #### Available crop_mode = ["single","multiple"]
        self.aten_list=[]
        self.depth_list=[]
        self.final_count_pack=[]
        self.digital_filter=["No_Comp_Vision_Filter",
                             "Threshold_Otsu",
                             "Temporal_median_filter",
                             "Spatial_Median_filter",
                             "Spatial_Bilateral_filter"]
        for folder_count in range(folders_numb):
            self.single_folder(folder_count)
            ################
            
            for dif_filt in self.digital_filter:
                if dif_filt=="Threshold_Otsu":
                    amp_img, self.bin_mask=self.fs.dual_otsu(self.grouped_array[0][0],single_data=True)
                    mod_grouped_array=self.grouped_array
                elif dif_filt=="Spatial_Median_filter":
                    mod_grouped_array, _=self.fs.multi_filter(self.grouped_array,
                                                                  feature_num=2,mode="Spatial_Median_filter")
                elif dif_filt=="Spatial_Bilateral_filter":
                    mod_grouped_array, _=self.fs.multi_filter(self.grouped_array,
                                                                  feature_num=2,mode="Spatial_Bilateral_filter")
                elif dif_filt=="Temporal_median_filter":
                    t_amp, _=self.fs.temporal(self.grouped_array,frame_hist=10,feature_num=0)
                    t_depth, _=self.fs.temporal(self.grouped_array,frame_hist=10,feature_num=2)
                    #from IPython import embed; embed()
                    #pass
                elif dif_filt=="No_Comp_Vision_Filter":
                    mod_grouped_array=self.grouped_array
                    self.bin_mask=np.ones(mod_grouped_array[0,0,:,:].shape,dtype="bool")

                for crop_label in crop_label_list:
                    roi_coord=self._roi_coord_mode(mode=mode,crop_label=crop_label)
                    if roi_coord!=0:
                        afv=self.amp_filt_val
                        if dif_filt=="Temporal_median_filter":
                            error_depth_array,amp_data,depth_data=self.temporal_analysis(t_amp,t_depth,roi_coord,crop_label,
                                                                                       low_amp_filter=True,amp_filt_val=afv)
                        elif dif_filt=="No_Comp_Vision_Filter":
                            error_depth_array,amp_data,depth_data=self.single_object_analysis(mod_grouped_array,
                                                                                            roi_coord,crop_label,
                                                                                            low_amp_filter=True,amp_filt_val=0)
                        else:
                            error_depth_array,amp_data,depth_data=self.single_object_analysis(mod_grouped_array,
                                                                                            roi_coord,crop_label,
                                                                                            low_amp_filter=True,amp_filt_val=afv)
                        m1=amp_data>afv
                        if m1.sum()!=0:                    
                            error_depth_array=error_depth_array[m1]
                            amp_data=amp_data[m1];      depth_data=depth_data[m1]

                            pack=[crop_label,self.exp_aten,self.exp_depth,
                            round(abs(error_depth_array).mean(),3), round(abs(error_depth_array).std(),3),
                            round(amp_data.mean(),1), round(amp_data.std(),1),
                            round(depth_data.mean(),3),round(depth_data.std(),3),
                            m1.sum(),amp_data.ravel(),error_depth_array.ravel(),depth_data.ravel(),dif_filt]
                            self.final_count_pack.append(pack)
    
    def temporal_analysis(self,t_amp,t_depth,
                          roi_coord,crop_label,
                          low_amp_filter=True,amp_filt_val=3):

        if low_amp_filter:
            amp_data_mask=t_amp[0,:,:]>amp_filt_val  
            masked_array_amp=self.fs.apply_mask_temporal(t_amp,amp_data_mask)
            masked_array_depth=self.fs.apply_mask_temporal(t_depth,amp_data_mask)
            
            if amp_data_mask.sum()==0:
                masked_array_amp=t_amp
                masked_array_depth=t_depth

        else:
            masked_array_amp=t_amp
            masked_array_depth=t_depth

        cropped_array_amp=self.cp.temporal_crop(masked_array_amp,roi_coord)
        cropped_array_depth=self.cp.temporal_crop(masked_array_depth,roi_coord)
        cropped_mask=self.cp.single_feature_crop(self.bin_mask,roi_coord)

        if cropped_mask.sum()==0:
            cropped_mask_array_amp=cropped_array_amp
            cropped_mask_array_depth=cropped_array_depth
        else:   
            cropped_mask_array_amp=self.fs.apply_mask_temporal(cropped_array_amp,cropped_mask)
            cropped_mask_array_depth=self.fs.apply_mask_temporal(cropped_array_depth,cropped_mask)

        depth_max=7496
        error_depth_array=self.fs.temporal_depth_check(cropped_mask_array_depth, self.exp_depth,depth_max=depth_max)
        #### Check Effects of error mask over depth array and amp_data
        amp_data=cropped_mask_array_amp
        depth_data=cropped_mask_array_depth

        return error_depth_array,amp_data,depth_data

    def single_object_analysis(self,grouped_array,roi_coord,crop_label,low_amp_filter=False,amp_filt_val=3):
        if low_amp_filter:
            amp_data_mask=grouped_array[0,0,:,:]>amp_filt_val   
            masked_array=self.fs.apply_mask(grouped_array,amp_data_mask)
        else:
            masked_array=grouped_array
        cropped_array=self.cp.multi_feature_crop(masked_array,roi_coord)
        cropped_mask=self.cp.single_feature_crop(self.bin_mask,roi_coord)
        if cropped_mask.sum()==0:
            cropped_masked_array=cropped_array
        else:   
            cropped_masked_array=self.fs.apply_mask(cropped_array,cropped_mask)

        depth_max=7496
        error_depth_array=self.fs.apply_depth_check(cropped_masked_array, self.exp_depth,depth_max=depth_max)
        #### Check Effects of error mask over depth array and amp_data
        amp_data=cropped_masked_array[:,0,:,:]
        depth_data=cropped_masked_array[:,2,:,:]
        if (self.exp_depth<(depth_max+100) and self.exp_depth>(depth_max-100)):
            phase_mask_low=cropped_masked_array[0,4,:,:]<(70)
            phase_mask_remm=np.logical_not(phase_mask_low)
            #from IPython import embed;embed()
            depth_data_low=self.fs.apply_mask(cropped_masked_array,phase_mask_low)[:,4,:,:]+7.496
            depth_data_high=self.fs.apply_mask(cropped_masked_array,phase_mask_remm)[:,4,:,:]
            depth_data=depth_data_low+depth_data_high
        elif self.exp_depth>depth_max:
            depth_data=depth_data+7.496
        return error_depth_array,amp_data,depth_data    

    def _apply_histogram(self,amp_data,error_depth_data,plot=False):
        self.amp_h=Main_Histogram(amp_data)
        self.amp_h.histo()
        self.err_d=Depth_Error_hist(self.amp_h.hist_bins)
        final_count=self.err_d.loop_mask(amp_data,error_depth_data)
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
    
    @staticmethod
    def histogram3D(amp_data,depth_data):
        a=Main_Histogram(amp_data,step=15)
        a.histo()
        a_del=np.where(a.count==0)
        a.hist_bins=np.delete(a.hist_bins,a_del)

        a_del2=np.where(a.hist_bins<19)
        a.hist_bins=np.delete(a.hist_bins,a_del2)

        depth_data=(depth_data*1000).astype("int")
        b=Main_Histogram(depth_data,step=25)
        b.histo()
        b_del=np.where(b.count==0)
        b.hist_bins=np.delete(b.hist_bins,b_del)

        # get bin pair of 'a'
        pair_a=[]
        pair_b=[]
        pair3d=[]
        for i in range(len(a.hist_bins)):
            if i+1<len(a.hist_bins):
                pair_a.append([a.hist_bins[i],a.hist_bins[i+1]])
        for ii in range(len(b.hist_bins)):
            if ii+1<len(b.hist_bins):
                pair_b.append([b.hist_bins[ii],b.hist_bins[ii+1]])
        for pa in pair_a:
            for pb in pair_b:
                amp_mask=np.logical_and(amp_data>=pa[0],amp_data<pa[1])
                depth_mask=np.logical_and(depth_data>=pb[0],depth_data<pb[1])
                final_mask=np.logical_and(amp_mask,depth_mask)
                mask_count=final_mask.sum()
                if mask_count>1:
                    pair3d.append([pa[0],pb[0],mask_count])
        hist3Darr=np.array(pair3d)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(hist3Darr[:,0],hist3Darr[:,1],hist3Darr[:,2])
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Count')
        plt.show()

    def read_saved_analysis(self,excel_filename):
        self.df=pd.read_excel(excel_filename, index_col=0)
        self.dfr=Dataframe_results(self.df)
        print("Available Column Names",self.dfr.label_names)

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
    
    def amp_depth_curve(self, final_count_pack,aten_filt,label_filt,mode="Amp",plot_save=False):
        import matplotlib.cm as cm
        #from IPython import embed; embed()
        #ax= fig.add_subplot(111,aspect='auto')
        fig=plt.figure()
        fig.set_size_inches(25, 14.18)
        ax1=fig.add_subplot(111,aspect='auto')
        count=0
        for num,pack in enumerate(final_count_pack):
            if pack[3]!=10:
                count=count+1
                label,aten,depth,err_m,err_st,amp_m,amp_std,depth_m,depth_std,amp_count,amp_all,error_all,depth_all=pack
                if (aten==aten_filt and label==label_filt):
                    name=label+"-"+aten+"-"+str(depth)
                    width=depth_std
                    if mode=="Amp":
                        center=[depth_m,amp_m]
                        height=amp_std
                    elif mode=="Error":
                        center=[depth_m,err_m]
                        height=err_st
                    
                    colors1 = cm.rainbow(np.linspace(0, 1, len(self.depth_list)))
                    #from IPython import embed;embed()
                    for dnum, delem in enumerate(self.depth_list):
                        if depth==delem:
                            color1=colors1[dnum]

                    ax1.scatter(center[0],center[1],c=color1,label=name+"_mean_center")
                    #ax1.scatter(depth_all,amp_all,c=color1,marker="x",label=name+"_scatter")
                    plt.errorbar(center[0], center[1], yerr=height, uplims=True, lolims=True,
                        label=name+"_mean_center")
                    #ellipse1 = ptc.Ellipse(center, width, height, color=color1, linewidth=2, fill=True, zorder=2,alpha=0.65)
                    #ax1.add_patch(ellipse1)
                    
        if count!=0:
            if mode=="Amp":
                ax1.set_ylabel('Amplitude')
                ax1.set_ylim(0, 800)
                start, end = ax1.get_ylim()
                ax1.yaxis.set_ticks(np.arange(start, end, 25))

            elif mode=="Error":
                ax1.set_ylabel('Depth Error[m]')
                ax1.set_ylim(0, 0.6)
                start, end = ax1.get_ylim()
                ax1.yaxis.set_ticks(np.arange(start, end, 0.020))

            ax1.set_xlabel('Experiment Depth[m]')
            ax1.set_xlim(0, 13)
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks(np.arange(start, end, 0.5))
            ax1.set_title("Depth vs "+ mode+" - Aten: "+aten_filt+" - Label: "+label_filt+
                          " - Amplitude Filter= "+str(self.amp_filt_val))
            ax1.legend()
            plt.grid(True)
            if plot_save:
                folder="d:/tof/outputs2/"+label_filt+"/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                filename="_ampfilt_"+str(self.amp_filt_val)+"_Depth_vs_"+mode+"_aten_"+aten_filt+"_label_"+label_filt+".png"
                filepath=folder+filename
                plt.savefig(filepath, dpi=300,format='png')
            
            #plt.show()

    def _roi_coord_mode(self,mode,crop_label):

        if self.exp_aten=="No_Optical_Filter":
            aten_num=0
        else:
            aten_num=1

        if mode=="object":
            roi_coord=self.get_crop_coord(crop_label=crop_label,aten_num=aten_num)
        elif mode=="patch":
            roi_coord=self.get_patch_coord(crop_label=crop_label,depth=self.exp_depth,aten_num=aten_num)
        return roi_coord     

    def _depth_aten_update(self):
        if self.exp_aten not in self.aten_list:
            self.aten_list.append(self.exp_aten)

        if self.exp_depth not in self.depth_list:
            self.depth_list.append(self.exp_depth)

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