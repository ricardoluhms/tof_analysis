import cv2
import numpy as np

#### Customized Libraries  - Vinicius
from filters.utils import filter
from filters.utils import edge_filter as edf

class Feature_show():
    def __init__(self):
        pass

    @staticmethod
    def std_window_show(window_name,array):
        cv2.namedWindow(window_name,cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window_name,array)
    
    @staticmethod
    def single_window(window_name,array):
        cv2.namedWindow(window_name,cv2.WINDOW_KEEPRATIO)
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

    @staticmethod
    def feature_show(grouped_array):
        fs=Feature_show()
        for i in range(grouped_array.shape[0]):
            fs.std_window_show("Amplitude",filter.norm(fs._check_exploding(grouped_array[i][0])))
            fs.std_window_show("Ambient",filter.norm(fs._check_exploding(grouped_array[i][1])))
            fs.std_window_show("Depth",filter.norm(fs._check_exploding(grouped_array[i][2])))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @staticmethod
    def get_otsu(grouped_array,frame_num,feature_num,crop_enhance=False):
        fs=Feature_show()
        frame_img=grouped_array[frame_num][feature_num]
        if crop_enhance:
            amp_mask_enhan=np.logical_and(frame_img>25,frame_img<80)
            frame_img[:,:][amp_mask_enhan]=80
        ##########################################################
        img=filter.norm(fs._check_exploding(frame_img))
        mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        mask=(mask==255)
        img=img*mask
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        return img, mask

    @staticmethod
    def dual_otsu(grouped_array,frame_num=0,feature_num=0,crop_enhance=False,single_data=False):
        ### raw frame
        fs=Feature_show()
        if single_data:
            frame_img=grouped_array
        else:
            frame_img=grouped_array[frame_num][feature_num]
        ### check exploding and normalize input
        img=filter.norm(fs._check_exploding(frame_img))
        ### Get first Threshold
        mask1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        ### First mask
        mask2=(mask1==255)
        ### Intermediary mask and remaining data for second threshold
        mask3=np.logical_not(mask2)
        img2=img*mask3
        ### Get Second Threshold
        mask4 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        ### Second mask
        mask5=(mask4==255)
        final_mask=np.logical_or(mask2,mask5)
        final_img=img*final_mask

        return final_img, final_mask


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

            self.dual_otsu(any_grouped_array,fram,0)
        #     ot_img, ot_mask=self.get_otsu(any_grouped_array,fram,0)
        #     #from IPython import embed; embed()
        #     if len(ot_img.shape)==3:
        #         ot_img=(ot_img[:,:,0]+ot_img[:,:,1]+ot_img[:,:,2])/3

        #     ot_mask=ot_mask.astype("int")*254
        #     ot_mask=ot_mask/ot_mask.max()
        #     otsu_amp_mask[fram]=ot_mask
        #     otsu_amp_out[fram]=ot_img
        #     ###Edge
        #     edge_depth[fram]=edge.apply(depth[fram,:,:]).reshape(a_shape[2],a_shape[3])
        #     edge_amp[fram]=edge.apply(amp[fram,:,:]).reshape(a_shape[2],a_shape[3])
        #     ###Error
        #     d_err_img=depth_error[fram]
        #     ### Amp Norm data
        #     raw_amp_img=amp[fram]
            
        #     amp_img_w=self.conv_weight(raw_amp_img)
        #     amp_img=self._check_exploding(raw_amp_img)
        #     amp_img2=self._check_exploding(amp_img_w)

        #     raw_amp_img=raw_amp_img/raw_amp_img.max()
        #     amp_img=amp_img/amp_img.max()
        #     amp_img2=amp_img2/amp_img2.max()

        #     ostu_edge=edge.apply(ot_img).reshape(a_shape[2],a_shape[3])
        #     ostu_mask_edge=edge.apply(ot_mask).reshape(a_shape[2],a_shape[3])

        #     self.std_window_show("Raw Amp Img",raw_amp_img)
        #     self.std_window_show("Check Amp Img",amp_img)
        #     self.std_window_show("Ostu Img",ot_img)
        #     self.std_window_show("Ostu Mask",ot_mask)
        #     self.std_window_show("Ostu Edge Img",ostu_edge)
        #     self.std_window_show("Ostu Edge Mask",ostu_mask_edge)

        #     key_pressed = cv2.waitKey(300) & 0xff
        #     if key_pressed in [32, ord('p')]:
        #         key_pressed = cv2.waitKey(0) & 0xff
        # cv2.destroyAllWindows()

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

        cv2.namedWindow(window_name,cv2.WINDOW_KEEPRATIO)
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