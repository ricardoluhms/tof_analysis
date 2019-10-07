import numpy as np
import os
from std_headers import Headers
def main():
    pass

class Input_data(Headers):
    def __init__(self,input_fld):
        #Texas Tof Model ['OPT8241']
        #self.main_header("Loading Input data","IPDT_001")
        #self.simple_header("Initial parameters for Texas Tof Model ['OPT8241']")
        #self.simple_header("device_resolution = 240x320 pixels")
        self.input_folder=input_fld
        self.simple_header(("Reading files from ... "+input_fld))
        #self.mx_frames=max_frames_to_read
        self.device_resol=[240,320] #### Adjust according to Kit Resolution - Standard is Texas Kit
        self.tt_points_per_frame=self.device_resol[0]*self.device_resol[1]
        self.path_list=[]
        self.file_dict={}
        self._file_type_dict()
        self.file_list(show_file=False)

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

    def _create_ipxl(self,x):
        i_pxl=(np.arange(0,x)).reshape(x,1)
        return i_pxl

    def _create_jpxl(self,i_pxl,y):
        for j in range(y):
            j_pxl=(np.ones(i_pxl.shape[0])*j).reshape(i_pxl.shape[0],1)
            i_j_ct=np.hstack((i_pxl,j_pxl))
            if j==0:
                i_j_pack=i_j_ct
            else:
                i_j_pack=np.vstack((i_j_pack,i_j_ct))
        return i_j_pack

    def _create_frame_track(self,i_j_pack,frames):
        for frame in range(frames):
            frame_ct=(np.ones(i_j_pack.shape[0])*frame).reshape(i_j_pack.shape[0],1)
            if frame==0:
                frame_i_j_pack=np.hstack((frame_ct,i_j_pack))
            else:
                frame_i_j_pack=np.vstack((frame_i_j_pack,np.hstack((frame_ct,i_j_pack))))
        return frame_i_j_pack.astype("int")

    def _create_coord(self,x,y,frames):
        i_pxl=self._create_ipxl(x)
        i_j_pack=self._create_jpxl(i_pxl,y)
        frame_i_j_pack=self._create_frame_track(i_j_pack,frames)
        return frame_i_j_pack

    def file_list(self,show_file=False):
        valid_files = ".bin"
        for file in os.listdir(self.input_folder):
            if file.endswith(valid_files):
                n_path=os.path.join(self.input_folder,file)
                if n_path not in self.path_list:
                    self.path_list.append(n_path)
        if show_file==True:
            self.simple_header("files in the selected folder: ")
            self.simple_header(self.input_folder)
            #print("path_list")
            #print(self.path_list)

    def _raw_single_file(self,file):
        _,b=os.path.split(file)
        filetype,_=os.path.splitext(b)
        #print("file name =",file," filetype =",filetype)
        if filetype in self.file_dict:
            dtype=self.file_dict[filetype]['dtype']
            fmap=self.file_dict[filetype]['map']
            data=np.fromfile(file,dtype=dtype)
        return data,filetype,fmap

    def _reshape_data(self,data,filetype,fmap):
        # Reshaping data
        if fmap==1:
            frames=int(len(data)/self.tt_points_per_frame)
            data=data.reshape(frames,self.device_resol[1],self.device_resol[0],fmap).swapaxes(-1,-3)
        elif fmap==4:
            frames=int(len(data)/(self.tt_points_per_frame*fmap))
            data=data.reshape(frames,fmap,self.device_resol[0],self.device_resol[1])
        return data

    def reshaped_single(self,file):
        data,filetype,fmap=self._raw_single_file(file)
        reshaped_data=self._reshape_data(data,filetype,fmap)
        return reshaped_data,filetype,fmap

    def reshaped_grouped(self):
        count=0
        self.file_list(self.input_folder) #### solve similiar problem
        for file in self.path_list:
            data,filetype,fmap=self._raw_single_file(file)
            reshaped_data=self._reshape_data(data,filetype,fmap)
            if count==0:
                #print("count start",count)
                dataGroup=reshaped_data
                #print("data shape",dataGroup.shape)
            else:
                dataGroup=np.hstack((dataGroup,reshaped_data))
            count+=1
            #print("data shape",dataGroup.shape,"count",count,"- file_type=",filetype)

        return dataGroup
    
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
    def reshaped_into_table(self,mx_frames=0):
        self.mx_frames=mx_frames
        self.main_header("Get Array","IPDT_001")
        array_group=self.reshaped_grouped()

        print("Initial shape", array_group.shape)
        if self.mx_frames>0:
            array_group=array_group[:self.mx_frames,:,:,:]
            print("Frame limited shape", array_group.shape)
        frames,_,y,x=array_group.shape
        
        array_table=self._rearrage_test(array_group)
        print("rearrange shape", array_table.shape)
        coord=self._create_coord(x,y,frames)
        self.table_array=np.hstack((coord,array_table))
        print("Conversion from Array into Frames")
        print("Assigned pixel coordinates and frames for each bin file ", self.table_array.shape)

class Mfolder():
    def __init__ (self,all_exp_folder):
        self.all_exp_folder=all_exp_folder
        self.folder_path_list=[]

    def swipe_folders(self):
        for folder in os.listdir(self.all_exp_folder):
            n_f_path=os.path.join(self.all_exp_folder,folder)       
            if os.path.isdir(n_f_path): 
                if n_f_path not in self.folder_path_list:
                    self.folder_path_list.append(n_f_path)

if __name__ == '__main__':
    main()