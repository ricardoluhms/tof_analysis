import numpy as np
import matplotlib.pyplot as plt

class Main_Histogram():
    def __init__(self,data,step=15):
        self.hist_min=data.min()
        self.hist_max=data.max()
        self.hist_bins=np.arange(self.hist_min,self.hist_max,step)
        self.data=data

    def histo(self):
        self.count, self.hist_bins=np.histogram(self.data,bins=self.hist_bins,density=False)

    def plot(self):
        # if self.hist_min==0:
        #     _ = plt.hist(self.count[1:], bins=self.hist_bins[1:])
        # else:
        #     _ = plt.hist(self.count, bins=self.hist_bins)
        #### Check shapes
        diff=self.hist_bins.shape[0]-self.count.shape[0]
        if diff!=0:
            self.hist_bins=self.hist_bins[:-diff]
        #### Remove bins with value 0
        a=np.where(self.count==0)
        new_count=np.delete(self.count,a)
        new_bins=np.delete(self.hist_bins,a)
        
        #### Check if one value most of the counts
        max_cod=new_count.max()/new_count.sum()
        print("max_cod= ", max_cod)
        if max_cod>0.6:
            b=np.where(self.count==new_count.max())
            new_count2=np.delete(new_count,b)
            new_bins2=np.delete(new_bins,b)
            #from IPython import embed; embed()
            plt.scatter(new_bins2,new_count2)

            plt.title("Histogram without max count."+
                      " Where max_count= "+str(new_count.max())+
                      " bin= "+str(new_bins[b[0]])+
                      " %="+str(int(max_cod*100)))    
            plt.show()
        

        plt.scatter(new_bins,new_count)       
        #_ = plt.hist(self.count, bins=self.hist_bins,density=False)
        plt.title("Histogram with max count")
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
            print ("loop_mask pair ",pair)
            final_mask=self._bins_mask(amp_data,pair[0],pair[1])
            print("loop_mask pair ",final_mask.sum())
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