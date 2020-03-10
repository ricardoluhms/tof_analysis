import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#matplotlib notebook


class pickle_tof():
    def __init__(self,filename):
        self.filename=filename
        
    def pickle_to_list(self):
        valid_values=[]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            #print(len(data))
            for datanum in range(len(data)):
                a=data[datanum].get('ids')
                if len(a)==0:
                    pass
                else:
                    for num in range(len(a)):
                        #from IPython import embed;embed()
                        if data[datanum].get('speed')[num][0]>1:
                            valid_values.append([data[datanum].get('speed')[num][0],data[datanum].get('ids')[num][0],
                                                 data[datanum].get('depth')[num][2]])
        return valid_values
    
    def transform_list_data(self,valid_values,flip=True,speed=10,flip_range=2):
        
        array=np.array(valid_values)
        array_mod=array.copy()
        if flip:
            for num in range(len(array)):
                if num+1<len(array):
                    if abs(array[num+1][2]-array[num][2])>flip_range and array[num+1][1]==array[num][1]:
                        flip=False
                    if flip:
                        array_mod[num][2]=array[num][2]+18.737
        
        self.df=pd.DataFrame(array_mod,columns=["speed","id","depth"])   
        self.ids=self.df['id'].drop_duplicates().tolist()
        self.depths=np.array(self.df['depth'].drop_duplicates())
        self.vel_std=np.ones(len(self.depths))*speed
        self.depth_ranges=np.linspace(round(self.depths.max()+1),0,num=(int(round(self.depths.max()+1)/0.5)))
        self.resumed=[]
        for i in self.ids:
            dfmask=self.df["id"]==i
            results=self.df[dfmask]
            for num,depth_r in enumerate(self.depth_ranges):
                if num+1<len(self.depth_ranges):
                    dfmask1=results["depth"]<=depth_r;  dfmask2=results["depth"]>self.depth_ranges[num+1]
                    dfmask3=dfmask1 & dfmask2
                    filt_results=results[dfmask3]
                    check=filt_results.empty
                    if check==False:
                    #from IPython import embed;embed()
                        speed_mean=filt_results["speed"].mean(); speed_std=filt_results["speed"].std()
                        self.resumed.append([self.depth_ranges[num+1],speed_mean,speed_std,i])
        self.dfresumed=pd.DataFrame(self.resumed,columns=["depth_range","vel_mean","vel_std","id"])
        
    def plot_data(self,title="No Title"):
        ax = plt.subplot(111)
        for i in self.ids:

            dfmask=self.dfresumed["id"]==i
            results=self.dfresumed[dfmask]
            plt.errorbar(results["depth_range"],results["vel_mean"],results["vel_std"],marker='o', label=i)
            #plt.plot(results["depth_range"], results["speed"],marker='o',linestyle='None', label=i)

        plt.plot(self.depths, self.vel_std, label="vel 10km/h")
        plt.xlabel("Depth [m]")
        plt.ylabel("Speed [km/h]")
        leg = plt.legend(loc='best',bbox_to_anchor=(1.20, 1.20))
        plt.title(title)
        plt.show()
            
def main():

    filename="C:/Users/ricar/Desktop/PUMA/TOF/pickle/infolist7T20.pkl"
    pck=pickle_tof(filename)
    pck_list=pck.pickle_to_list()
    pck.transform_list_data(pck_list,flip=True,speed=10)
    pck.plot_data("IT-5 FPS-60 Frame_Analysis-7 Threshold-20")
