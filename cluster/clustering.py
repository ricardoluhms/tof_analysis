#### Image and Array Operations Libraries
import numpy as np
import cv2
#### Folder and data libraries
import os
#### Plot Libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as ptc

#### Customized Libraries - Luhm
from std_headers import Headers
from input_data import Input_data
from feature_selection import Feature_selection
#### Customized Libraries - Vinicius

def main():
    folder_path='D:/Codes/Videos_tof/Experiments_Vel/Exp_2_Pan_90_Tilt_0_ilum_y_reflex_y_dist_1512'
    #folder_path='D:/Codes/Videos_tof/Experiments_Vel/Exp_0_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512'
    idt=Input_data(folder_path)
    #idt.main_header("Start Main","0")
    #idt.main_header("Complete Extraction of ToF Data from Texas OPT8241","00")
    #grouped_array=idt.reshaped_grouped()
    idt.reshaped_into_table(mx_frames=3)


    fs=Feature_selection(idt.table_array)
    fs.cov_matrix(fs.std_array) ## start full cov matrix
    #fs.plot_heatmap() ## start plot full cov matrix heatmap
    #fs.find_relevant() ## show most relevant features
    #columns_list=[0,1,2,8,9,10,11]
    columns_list=[0,8,9,10,11] #### Selection of relevant columns features befora PCA analysis
    fs.remove_column(columns_list)
    fs.get_eigen()
    fs.pca_list()
    #fs.plot_heatmap()
    cm=Cluster_manag(10,fs.std_array,fs.pca_pair_list[0])
    cm.start_all()
    for _ in range(10):
        cm.update_euclid_darray()
        cm.update_clust_resp()
        cm.update_clusters_center()


    #cm.cluster_resp_data()

    from IPython import embed; embed()

class Cluster_oper(Headers):
    def __init__(self):
        pass

    def calc_vector(self,cluster_coord,point_center):
        diff=point_center-cluster_coord
        xd=diff[0]
        yd=diff[1]
        if xd==0:
            xd=1/(10**50)
        return xd,yd

    def calc_distance(self,xd,yd):
        d=((xd**2)+(yd**2))**(1/2)
        return d

    def calc_quad(self,xd,yd):
        if ((xd>=0) and (yd>=0)):
            q=1
        elif ((xd<0) and (yd>=0)):
            q=2
        elif ((xd<0) and (yd<0)):
            q=3
        elif ((xd>=0) and (yd<0)):
            q=4
        return q
    
    def calc_thet(self,xd,yd,q):
        thet=np.arctan(yd/xd)*180/np.pi
        if q==2:
            thet=thet+180
        elif q==3:
            thet=thet+180
        elif q==4:
            thet=thet+360
        return thet

    def get_diff_series(self,cluster_coord,pt_center):
        xd,yd=self.calc_vector(cluster_coord,pt_center)
        q=self.calc_quad(xd,yd)
        thet=self.calc_thet(xd,yd,q)
        d=self.calc_distance(xd,yd)
        diff_series=np.array([xd,yd,thet,d,int(q)])
        return diff_series

class Cluster_data(Headers):
    def __init__(self,cluster_name,cluster_number,pca_std_array,cluster_center_input):
        self.samples_number=pca_std_array.shape[0]
        self.c_name=cluster_name
        self.cluster_coord=np.array([cluster_center_input])
         # center is [x,y]

    def start_single_euclid_darray(self):
        self.diff_array=np.ones((self.samples_number,5))
        return self.diff_array

    def start_single_resp_matrix(self):
        self.indv_resp_matrix=np.ones((self.samples_number,1))
        return self.indv_resp_matrix
    
    def start_single_inv_euclid_d_matrix(self):
        self.inv_euclid_d_matrix=np.zeros((self.samples_number,1))
        return self.inv_euclid_d_matrix

class Cluster_manag(Cluster_data,Cluster_oper,Headers):
    def __init__(self,total_clusters,std_table_array,pca_weight_list_item):
        ### Starting info to manage All Cluster Data
        self.clust_name_list=[]
        self.tt_clusters=total_clusters
               
        ### Store starting info of the input samples to be clusterized
        ##### Get PCA's Get Weights
        self.pca_std_array=std_table_array.dot(pca_weight_list_item)
        self.samples_number=self.pca_std_array.shape[0]

        ### Initial Status of Cluster Numbers
        self.reduce_clust=False
        self.valid_prob=0.30
    
    def start_all(self):
        name="Start Clusters, Euclidian Distances, Cluster Resposability Matrix and Inverse of Euclideand Dist."
        self.main_header(name,"CL_MNG_001")
        self.start_cluster_names() #1
        self.start_centers() #2
        self.start_euclid_darray() #3
        self.start_resp_array() #4
        self.start_inv_euclid_darray() #5

    def start_cluster_names(self):
        name=("Starting Cluster Names: Will be used to track cluster status and properties")
        self.second_header(name,1,1)
        for clust in range(self.tt_clusters):
            if clust>10:
                name="Cluster - 0"+str(clust)
            else:
                name="Cluster - 00"+str(clust)
            self.clust_name_list.append(name)

    def start_centers(self):
        name=("Starting Cluster Center: Define Starting for each Cluster Center")
        self.second_header(name,1,2)
        self.c_centers=np.zeros((self.tt_clusters,2))
        min_PCA0=self.pca_std_array[:,0].min()
        max_PCA0=self.pca_std_array[:,0].max()
        min_PCA1=self.pca_std_array[:,1].min()
        max_PCA1=self.pca_std_array[:,1].max()
        for clust in range(self.tt_clusters):
            if clust==0:
                self.c_centers[clust]=np.array([max_PCA0,max_PCA1])
            elif clust==1:
                self.c_centers[clust]=np.array([min_PCA0,min_PCA1])
            elif clust==2:
                self.c_centers[clust]=np.array([min_PCA0,max_PCA1])
            elif clust==3:
                self.c_centers[clust]=np.array([max_PCA0,min_PCA1])
            else:
                self.c_centers[clust]=self.pca_std_array[np.random.choice(self.samples_number)]

    def start_euclid_darray(self):
        name=("Starting Complete Diff Arrays: Calculate and Store Euclidean Distances of Clusters"+
        "and Other parameters such [xd,yd,thet,euclid dist,quad]")
        self.second_header(name,1,3)
        for clust in range(self.tt_clusters):
            if clust==0:
                self.euclid_darray=np.array([self.start_single_euclid_darray()])
            else:
                self.euclid_darray=np.vstack((
                    self.euclid_darray,
                    np.array([self.start_single_euclid_darray()])))

    def start_resp_array(self):
        name=("Starting a Complete Array - Cluster Responsability : Shape => rows = samples, columns = clusters")
        self.second_header(name,1,4)
        for clust in range(self.tt_clusters):
            if clust==0:
                self.clust_resp_array=self.start_single_resp_matrix()
            else:
                self.clust_resp_array=np.hstack((
                    self.clust_resp_array,
                    self.start_single_resp_matrix()))

    def start_inv_euclid_darray(self):
        name=("Starting a Complete Array - Inverse of the Euclidean Distance: Shape => rows = samples, columns = clusters")
        self.second_header(name,1,5)
        for clust in range(self.tt_clusters):
            if clust==0:
                self.inv_euclid_d_array=self.start_single_inv_euclid_d_matrix()
            else:
                self.inv_euclid_d_array=np.hstack((
                    self.inv_euclid_d_array,
                    self.start_single_inv_euclid_d_matrix()))
  
    def update_euclid_darray(self):
        name=("Updating Euclidean Distances Array and the Inverse of its distance")
        self.second_header(name,1,6)
        for clust in range(self.tt_clusters):
            for sample in range(self.samples_number):
                ##### 
                if self.clust_resp_array[sample,clust]>self.valid_prob :
                    self.euclid_darray[clust,sample,:]=self.get_diff_series(
                        self.c_centers[clust], self.pca_std_array[sample]
                        )
                    if self.euclid_darray[clust,sample,3]==0:
                        self.euclid_darray[clust,sample,3]=1/(10**30)
                    self.inv_euclid_d_array[sample,clust]=(1/self.euclid_darray[clust,sample,3])
                else:
                    self.inv_euclid_d_array[sample,clust]=0

    def update_clust_resp(self):
        name=("Updating a Complete Array - Responsability Matrix: "+
        "Update Responsabilities values for each Cluster")
        self.second_header(name,1,7)

        self.potential_resp=(self.inv_euclid_d_array/
                            self.inv_euclid_d_array.sum(axis=1, keepdims=True))

        for sample in range(self.potential_resp.shape[0]):
            self.clust_resp_array[sample,:]=(
                self.potential_resp[sample,:]/
                self.potential_resp[sample,:].max()
            )
        self.get_points_in_cluster()

        #### div1 is similar to ttpoints_in_cluster but div1 will be modified to avoid division by 0
        self.division=self.clust_resp_mask.sum(axis=0, keepdims=True).T
        self.division[self.division==0]=1

    def get_points_in_cluster(self):
        self.third_header("#Validate Respons. and count valid points in cluster",1,7,1)
        #### Mask1 - Ask: Are there respons. below the valid probability set initialy
        #### Answer: True=Yes, False=No
        mask1=self.clust_resp_array[:,:]<self.valid_prob 
        #### Mask2 - Ask: Are there respons. equal or above the valid probability set initialy
        #### Answer: True=Yes, False=No
        self.clust_resp_mask=np.logical_not(mask1)
        #### ** clust_resp_mask and clust_resp_array are almost the same except for:
        #### clust_resp_mask is boolean array
        #### clust_resp_array is float array

        #### Sum True Values of Mask2 - Return the number of points which probably belongs to the cluster
        #### This probability will be validated later
        self.ttpoints_in_cluster=self.clust_resp_mask.sum(axis=0) #### Sum true values
        #### Update Responsability Matrix part 1 
        #### - All values below the threshold will be 0, otherwise it will 1       
        self.clust_resp_array[mask1]=0
        #self.clust_resp_array[self.clust_resp_mask]=1 ****  

    def cluster_resp_data(self):
        self.second_header("#Calculate Initial Responsabilities for each Cluster",1,8)
       
        self.quad_resp=(self.euclid_darray[:,:,4].T*self.clust_resp_mask)
        ### Find in which quadrant is located most of the data. 
        self.count_quadrant()
        self.check_main_quadrants()
        self.group_data_mean_and_std()
        self.cluster_dcheck()

        # # V2 and V3 DATA row=cluster, column=[theta_mean,theta_std,rd_mean,rd_std]
        # V2=group_data_mean_and_std(M,self.quad_resp,
        # quad_main,self.euclid_darray,self.clust_resp_array)
        #  # V2 Main Quad
        # V3=group_data_mean_and_std(M,self.quad_resp,
        # quad_sec,self.euclid_darray,self.clust_resp_array) # V3 Sec Quad
        # V4=np.hstack((V2,V3))
        # V5=np.zeros((V4.shape[0],1)) #### added to solve cluster_str_indx issue
        # V4=np.hstack((V4,V5)) 

        # main_map,main_map2=cluster_relation_map(self.euclid_darray,M,V4) #cluster_relation_map=>full_map,closest_map

        # self.clust_resp_array=cluster_resp_loop(main_map,M,quad_main,quad_sec,V4,self.euclid_darray,self.quadrant_sum,self.clust_resp_array)
        # self.clust_resp_array=cluster_resp_loop(main_map2,M,quad_main,quad_sec,V4,self.euclid_darray,self.quadrant_sum,self.clust_resp_array)

        # mask=np.zeros((self.clust_resp_array.shape))
        # for sample in range(self.clust_resp_array.shape[0]):
        #     mask[sample,:]=self.euclid_darray[sample,:,3]==self.euclid_darray[sample,:,3].min()
        
        # for sample in range(self.clust_resp_array.shape[0]):
        #     if self.clust_resp_array[sample,:].sum()>1:
        #         self.clust_resp_array[sample,:]=mask[sample,:].astype(int)


        #return self.clust_resp_array,V4

    def count_quadrant(self):
        self.third_header("#Count points per cluster quadrant",1,8,1)
        for q in range(4):
            qn=q+1
            if q==0:
                self.quadrant_sum=(self.quad_resp[:,:]==qn).sum(axis=0)
            else:
                self.quadrant_sum=np.vstack(
                    (self.quadrant_sum, (self.quad_resp[:,:]==qn).sum(axis=0)
                    ))

    def check_main_quadrants(self):
        self.third_header("Find Quadrant with most relevant data",1,8,2)
        quad_mult=np.array([[1],[2],[3],[4]])
        q_sum=self.quadrant_sum
        for quad in range(4):
            quad_prio=q_sum/q_sum.max(axis=0)
            mask1=quad_prio==1
            quad_main=(mask1*quad_mult).sum(axis=0)
            if quad==0:
                self.quad_main=quad_main
            else:
                self.quad_main=np.vstack((self.quad_main,quad_main))
            q_sum=q_sum*np.logical_not(mask1)

    def group_data_mean_and_std(self):
        self.third_header("Calculate Mean and Std",1,8,3)
        shap=self.euclid_darray[:,:,2].T.shape
        thet_w_array=self.euclid_darray[:,:,2].T*self.clust_resp_array
        diag_w_array=self.euclid_darray[:,:,3].T*self.clust_resp_array
        quad_w_array=self.euclid_darray[:,:,4].T

        for num,series in enumerate(self.quad_main):
            mask1=np.zeros(shap)
            for sample in range(shap[0]):
                mask1[sample,:]=series
            #####
            mask2=np.equal(quad_w_array,mask1)
            thet_qw_arr=thet_w_array*mask2
            diag_qw_arr=diag_w_array*mask2
            #####
            thet_mean=thet_qw_arr.mean(axis=0)
            thet_std=thet_qw_arr.std(axis=0)
            diag_mean=diag_qw_arr.mean(axis=0)
            diag_std=diag_qw_arr.std(axis=0)
            #####
            if num==0:
                thet_stack_mean=thet_mean
                thet_stack_std=thet_std
                diag_stack_mean=diag_mean
                diag_stack_std=diag_std
            else:
                thet_stack_mean=np.vstack((thet_stack_mean,thet_mean))
                thet_stack_std=np.vstack((thet_stack_std,thet_std))
                diag_stack_mean=np.vstack((diag_stack_mean,diag_mean))
                diag_stack_std=np.vstack((diag_stack_std,diag_std))
            
        self.full_stack=np.vstack((
                            np.array([thet_stack_mean]), np.array([thet_stack_std]),
                            np.array([diag_stack_mean]), np.array([diag_stack_std])
                                ))

    def cluster_dcheck(self):
        conflict_list=[]
        for k in range(self.c_centers.shape[0]):
            for k2 in range(self.c_centers.shape[0]-1):
                kn=k2+1
                if k<kn:
                    A_to_B=self.get_diff_series(self.c_centers[k], self.c_centers[kn])
                    B_to_A=self.get_diff_series(self.c_centers[kn], self.c_centers[k])
                    A_d_mean=self.full_stack[2][A_to_B[4]][k]
                    A_d_std=self.full_stack[3][A_to_B[4]][k]
                    B_d_mean=self.full_stack[2][B_to_A[4]][kn]
                    B_d_std=self.full_stack[3][B_to_A[4]][kn]
                    dAB=np.array([A_d_mean,A_d_std,B_d_mean,B_d_std]).sum()

                    if A_to_B[3]>dAB:
                        d_check=False # distance between cluster is higher than sum of each radius
                    else:
                        d_check=True #potential cluster conflict
                        conflict_list.append(
                            [k,kn,
                             A_to_B[4],B_to_A[4],
                             A_d_mean,A_d_std,
                             B_d_mean,B_d_std,
                             dAB,A_to_B[3],
                             d_check
                        ])
        #conflict_list format [ # "cluster k","cluster kn",
                        # "quad clust k", "quad clust kn",
                        # "prob. radius (mean+1*std) of k in quad", 
                        # "prob. radius (mean+1*std) of kn in quad",
                        # "Sum of the prob. radius of k and kn", "true distance between k an kn",
                        # "distance check -True/False"]
        self.conflict_dmatrix=np.array(conflict_list)

    def update_clusters_center(self):
        self.c_centers=self.clust_resp_array.T.dot(self.pca_std_array)/self.division

    def plot_cluster_scatter(self,i,M,R2,V4,Y,color_map,clt_names_ls,show_plots=True):
        ####
        fig=plt.figure()
        ax=fig.add_subplot(111,aspect='auto')
        ### cmap = ListedColormap(color_map)

        for clt in range(self.c_centers):
            #place cluster name in its current M coordinate
            center=self.c_centers[clt]
            plt.text(center[0], center[1], 
                     self.clust_name_list[clt], bbox=dict(facecolor='red', alpha=0.5))
            # draw inclined elipse
            width=V4[clt,2]+V4[clt,3]
            height=V4[clt,6]+V4[clt,7]
            angle=V4[clt,0]
            ellipse = ptc.Ellipse(center, width, height,
                        angle=angle,color=color_map[clt], linewidth=2, fill=False, zorder=2)
            ax.add_patch(ellipse)
            mask=R2[:,clt]==1
            X_pca=Y[:,0][mask]
            Y_pca=Y[:,1][mask]
            ax.scatter(X_pca, Y_pca,c=tuple(color_map[clt]))
            if clt==M.shape[0]:
                pass    


if __name__ == '__main__':
    main()    
        


