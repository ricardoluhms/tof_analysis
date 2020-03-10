import cv2
import numpy as np

#### Customized Libraries  - Vinicius
from filters.utils import filter
from viewer.utils import pcv


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
    def feature_show(grouped_array):
        fs=Feature_show()
        for i in range(grouped_array.shape[0]):
            fs.std_window_show("Amplitude",filter.norm(fs._check_exploding(grouped_array[i][0])))
            fs.std_window_show("Ambient",filter.norm(fs._check_exploding(grouped_array[i][1])))
            fs.std_window_show("Depth",filter.norm(fs._check_exploding(grouped_array[i][2])))
            cv2.waitKey(0)
        cv2.destroyAllWindows()

class Crops(Feature_show,pcv):

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
    
    def crop_img(self,img_o,crop_obj=1,std_label=True,exp_depth=1):
        global img, refPt, cropping
        
        fs=Feature_show()
        refPt=[]
        roi_crop=[]
        cropping = False
        if img_o.dtype!="uint8" or len(img_o.shape)<=2:
            img_o=255*img_o/img_o.max()
            img_o=self.heat_map(img_o)
        
        img = img_o.copy()
        #img=self._check_exploding(img)
        img_clone = img.copy()
        roi_img = img.copy()
        window_name=("Crop object")

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
            i_min,j_min,i_max,j_max=roi_coord

        cv2.destroyAllWindows()
        crop_label=self.add_label(label_num=crop_obj,std_label=std_label)
        color=(255,255,255)
        roi_img=cv2.rectangle(roi_img, (roi_coord[0], roi_coord[1]), (roi_coord[2], roi_coord[3]), color, 1)
        cv2.putText(roi_img, crop_label, (roi_coord[0], roi_coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, lineType=cv2.LINE_AA)

        roi_crop=[i_min,j_min,i_max,j_max,crop_label,exp_depth]
        return roi_crop, roi_img

    def _crop_coord_detect(self,refPt,img):
        x_max=max(refPt[0][0],refPt[1][0])
        x_min=min(refPt[0][0],refPt[1][0])
        y_max=max(refPt[0][1],refPt[1][1])
        y_min=min(refPt[0][1],refPt[1][1])
        roi = img[y_min:y_max, x_min:x_max]
        roi_coord = [x_min,y_min,x_max,y_max]
        return roi, roi_coord

    @staticmethod
    def add_label(label_num=1,std_label=True):
        if std_label==False:
            label=input("Set Label Name for cropped region:")
        else:
            label="std_label-"+str(label_num)
        return label
    
    @staticmethod
    def get_bbox(img,pixel_area=100):
        _, connec, stats, centroids = cv2.connectedComponentsWithStats(img)
        #from IPython import embed; embed()
        labels = np.unique(connec)
        bndboxes = np.zeros((labels.shape[0]-1, 4), dtype='uint16')
        for c in range(1,labels.shape[0]):
            coord_y, coord_x = np.where(connec==c)
            bndboxes[c-1,:] = np.array([coord_x.min(), coord_y.min(), coord_x.max(), coord_y.max()], dtype='uint16')
        bndboxes = bndboxes[(bndboxes[:,2]-bndboxes[:,0])*(bndboxes[:,3]-bndboxes[:,1])>pixel_area, :]
        bndboxes=Crop().bbox_in_bbox(bndboxes)
        return bndboxes
    
    @staticmethod
    def bbox_in_bbox(bndboxes):
        bboxes_in_bbox=[]
        true_bbox=[]
        for box1 in range(bndboxes.shape[0]):
            for num in range((bndboxes.shape[0])-1):
                box2=num+1
                if box1<box2:
                    bc1=bndboxes[box1]
                    bc2=bndboxes[box2]
                    
                    cond1=(bc1[0]<=bc2[0] and bc1[1]<=bc2[1])
                    cond2=(bc1[2]>=bc2[2] and bc1[3]>=bc2[3])
                    if (cond1 and cond2):
                        bboxes_in_bbox.append(box2)

        for box in range(bndboxes.shape[0]):
            if box not in bboxes_in_bbox:
                true_bbox.append(box)
        bndboxes=bndboxes[true_bbox,:]
        return bndboxes

    @staticmethod
    def show_bbox(img,bndboxes,bbox_rename=False):
        bbox_names=[]
        z_bndboxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bbox_img=z_bndboxes.copy()
        #z_bndboxes = filter.norm(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        pixels = 20
        for obj,bndbox in enumerate(bndboxes):

            color=int(np.array(255/(1+(obj/2))).astype("uint8"))
            z_bndboxes = cv2.rectangle(z_bndboxes, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (255,color,color), 1)
            text="blob_"+str(obj)
            bbox_names.append(text)
            cv2.putText(z_bndboxes, text, (bndbox[0], bndbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,color,color), lineType=cv2.LINE_AA)
        cv2.namedWindow('z_bndboxes',cv2.WINDOW_KEEPRATIO)
        cv2.imshow('z_bndboxes', z_bndboxes)
        cv2.waitKey(1)
        if bbox_rename:
            bbox_img, bbox_names=Crop().rename_bbox(bbox_img,bbox_names,bndboxes)
            cv2.namedWindow('rename_bndboxes',cv2.WINDOW_KEEPRATIO)
            
        else:
            bbox_img=z_bndboxes	
        cv2.destroyAllWindows()
        return bbox_img, bbox_names
        
    @staticmethod
    def rename_bbox(bbox_img,bbox_names,bndboxes):
        cv2.namedWindow('rename_bndboxes',cv2.WINDOW_KEEPRATIO)
        for num,bbox_name in enumerate(bbox_names):
            color=int(np.array(255/(1+(num/2))).astype("uint8"))
            cv2.imshow('rename_bndboxes', bbox_img)
            print("Press any key after adjusting windows")
            cv2.waitKey(1)
            print("#### Rename ####")
            name="Rename BBox: "+bbox_name+" into: "
            bbox_name=input(name)
            bbox_names[num]=bbox_name
            z_bndboxes = cv2.rectangle(bbox_img, 
                                        (bndboxes[num][0], bndboxes[num][1]), (bndboxes[num][2], bndboxes[num][3]), 
                                        (255,color,color), 1)
            cv2.putText(bbox_img, bbox_name, (bndboxes[num][0], bndboxes[num][1]),
                
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,color,color), lineType=cv2.LINE_AA)
            cv2.imshow('rename_bndboxes', bbox_img)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        return bbox_img, bbox_names
    
    @staticmethod
    def single_box_error(img,mask,error,bndbox,pixels=9,show=False,bbox_num=0):
        
        cropped=img[bndbox[0]:bndbox[2],bndbox[1]:bndbox[3]]
        cropped_error=error[bndbox[0]:bndbox[2],bndbox[1]:bndbox[3]]
        cropped_mask=mask[bndbox[0]:bndbox[2],bndbox[1]:bndbox[3]]

        z_bndboxes = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        for y in range((bndbox[3]-bndbox[1])//pixels):
            y_0 = (y)*pixels
            y_1 = (y+1)*pixels
            pixels_error = []
            for x in range((bndbox[2]-bndbox[0])//pixels):
                x_0 = (x)*pixels
                x_1 = (x+1)*pixels
                pixels_mask = np.zeros(z_bndboxes.shape[:2], dtype='uint8')
                pixels_mask[y_0:y_1, x_0:x_1] = 1
                pixels_error.append(cropped_error[np.bitwise_and(mask!=0, pixels_mask!=0)].mean())
                z_bndboxes = cv2.rectangle(z_bndboxes, (x_0, y_0), (x_1, y_1), (255,0,0), 0.5)
            print(pixels_error)
            print('-'*90)
        print('*'*90)
        if show:
            cv2.namedWindow('z_bndboxes-blob_'+str(bbox_num),cv2.WINDOW_KEEPRATIO)
            cv2.imshow('z_bndboxes-blob_'+str(bbox_num), z_bndboxes)
        return pixels_error
    

        # error = (depth-1.013)
        # # error = (depth-3.009)
        # # error = (depth-4.834)
        # mean = depth[mask!=0].mean(); std = depth[mask!=0].std()
        # measure = depth[np.bitwise_and(mask!=0,np.bitwise_and(depth>=mean-2*std, depth<=mean+2*std))]
        # print('measure:%f+-%f [m]'%(measure.mean(),measure.std()))
        # print('  error:%f+-%f [m]'%(error[mask!=0].mean(),error[mask!=0].std()))

        # ===============================================================
        # 			drawing squares and computing error per square
        # ===============================================================
        z_bndboxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #z_bndboxes = filter.norm(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        pixels = 20
        for obj,bndbox in enumerate(bndboxes):

            color=int(np.array(255/(1+(obj/2))).astype("uint8"))
            z_bndboxes = cv2.rectangle(z_bndboxes, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (255,color,color), 1)
            text="blob_"+str(obj)
            cv2.putText(z_bndboxes, text, (bndbox[0], bndbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,color,color), lineType=cv2.LINE_AA)
            for y in range((bndbox[3]-bndbox[1])//pixels):
                y_0 = (y)*pixels + bndbox[1]
                y_1 = (y+1)*pixels + bndbox[1]
                pixels_error = []
                for x in range((bndbox[2]-bndbox[0])//pixels):
                    x_0 = (x)*pixels + bndbox[0]
                    x_1 = (x+1)*pixels + bndbox[0] 
                    pixels_mask = np.zeros(z_bndboxes.shape[:2], dtype='uint8')
                    pixels_mask[y_0:y_1, x_0:x_1] = 1
                    pixels_error.append(error[np.bitwise_and(mask!=0, pixels_mask!=0)].mean())
                    # z_bndboxes = cv2.putText(z_bndboxes, '%.4f'%pixels_error, (x_0, y_0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0))
                    z_bndboxes = cv2.rectangle(z_bndboxes, (x_0, y_0), (x_1, y_1), (255,0,0), 1)
                print(pixels_error)
                print('-'*90)
            print('*'*90)
        cv2.namedWindow('z_bndboxes',cv2.WINDOW_KEEPRATIO)
        cv2.imshow('z_bndboxes', z_bndboxes)