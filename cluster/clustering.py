#### Image and Array Operations Libraries
import numpy as np
import cv2
#### Folder and data libraries
import os
#### Plot Libraries
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as ptc

#### Machine Learning and Data Mining Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#### Customized Libraries
from std_headers import Headers
from input_data import Input_data

def main():
    folder_path='D:/Codes/Videos_tof/Experiments_Vel/Exp_2_Pan_90_Tilt_0_ilum_y_reflex_y_dist_1512'
    #folder_path='D:/Codes/Videos_tof/Experiments_Vel/Exp_0_Pan_90_Tilt_0_ilum_y_reflex_n_dist_1512'
    idt=Input_data(folder_path)
    idt.main_header("Start Main","0")
    idt.main_header("Complete Extraction of ToF Data from Texas OPT8241","00")
    #grouped_array=idt.reshaped_grouped()
    idt.reshaped_into_table()

    from IPython import embed; embed()

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
    cm.update_complete_diff_array()
    cm.update_clust_resp()

    from IPython import embed; embed()

class Feature_selection(Input_data,Headers):
    def __init__(self,array):
        print("Table Array Shape",array.shape)
        self.raw_array=array
        self.std_array= StandardScaler().fit_transform(self.raw_array)
        self.titles = ["frame", "i_pxl", "j_pxl", 
                        "Amp",  "Amb", "Depth", "Dist",
                        "Phase",  "PC_x", "PC_y", "PC_z","PC_i"
                        ]
        self.main_header("Starting Feature Selection","002")

    # Step 2 - Get Covariance Matrix
    def cov_matrix(self,std_array):
        self.main_header("Standard Cov. Matrix","003")
        self.cov_mat = np.cov(std_array.T)

    # Step 3 - Plot Cov Matrix as Heat Map 
    def plot_heatmap(self):
        self.main_header("Cov. Matrix Plot","004")
        fig, ax = plt.subplots()
        _ = ax.imshow(self.cov_mat)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(self.titles)))
        ax.set_yticks(np.arange(len(self.titles)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.titles)
        ax.set_yticklabels(self.titles)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
            # Loop over data dimensions and create text annotations.
        for i in range(len(self.titles)):
            for j in range(len(self.titles)):
                text = ax.text(j, i, np.round(self.cov_mat[i, j],decimals=1),
                            ha="center", va="center", color="w")

        ax.set_title("Cov Matrix Std")
        fig.tight_layout()
        plt.show()

    # Step 4 Find Relevant Features from Data
    def find_relevant(self):
        self.main_header("Find Relevant Cov Mat Features","005")
        feature_value=[]
        cov_mat2=np.abs(self.cov_mat)
        cov_mat2[cov_mat2>=1]=0
        total=cov_mat2.sum()
        cov_mat2=cov_mat2*100/total
        self.plot_heatmap()
        for feature in range(cov_mat2.shape[0]):
            p_sum=cov_mat2[:,feature].sum()
            feature_value.append(p_sum)
            print("feature-",self.titles[feature],feature," value",p_sum)
        print("-"*100)
        self.feature_relevant_values=dict(zip(self.titles,feature_value))

    # Step 5 Remove non relevant colums based on lists
    ### columns_list must be integer only
    ### columns_list can be non sequential
    def remove_column(self,columns_list):

        self.main_header("Revome Feature Columns Based on List","006")
        self.std_array=np.delete(self.std_array,columns_list,1)
        self.cov_matrix(self.std_array)
        indexes = columns_list
        for index in sorted(indexes, reverse=True):
            del self.titles[index]

    # Step 6 Get EigenVector and EigenValues from Selected Features based on the Cov Matrix
    def get_eigen(self):
        self.main_header("Get EigenVector and EigenValues from Selected Features","007")
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov_mat)
        self.eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:,i]) for i in range(len(self.eig_vals))]
        self.eig_pairs.sort()
        self.eig_pairs.reverse()

    # Step 7 Create Translation Matrix for Input Array to provide best PCA combinations
    def pca_list(self,plot_limit=1):
        self.main_header("Create Translation Matrix for Input Array to provide best PCA combinations","008")
        self.pca_pair_list=[]
        self.mtx_lst_PCA_name=[]
        for e_pair in range(len(self.eig_pairs)):
            while len(self.pca_pair_list)<=plot_limit:
                p1=e_pair
                p2=e_pair+1
                p1_name="Principal Component "+str(p1)
                p2_name="Principal Component "+str(p2)
                if (e_pair+1)<=len(self.eig_pairs):
                    matrix_w = np.hstack(
                        (self.eig_pairs[p1][1].reshape(self.eig_pairs[0][1].shape[0],1 ),
                            self.eig_pairs[p2][1].reshape(
                        self.eig_pairs[0][1].shape[0],1)))
                    self.pca_pair_list.append(matrix_w)
                    self.mtx_lst_PCA_name.append([p1_name,p2_name])

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

    def start_single_diff_array(self):
        self.diff_array=np.ones((self.samples_number,5))
        return self.diff_array

    def start_single_resp_matrix(self):
        self.indv_resp_matrix=np.ones((self.samples_number,1))
        return self.indv_resp_matrix
    
    def start_single_inv_euclid_d_matrix(self):
        self.inv_euclid_d_matrix=np.zeros((self.samples_number,1))
        return self.inv_euclid_d_matrix

class Cluster_manag(Cluster_data,Cluster_oper,Headers):
    def __init__(self,total_clusters,std_array,pca_weight_list_item):
        ### Starting info to manage All Cluster Data
        self.clust_name_list=[]
        self.tt_clusters=total_clusters
               
        ### Store starting info of the input samples to be clusterized
        ##### Get PCA's Get Weights
        self.pca_std_array=std_array.dot(pca_weight_list_item)
        self.samples_number=self.pca_std_array.shape[0]

        ### Initial Status of Cluster Numbers
        self.reduce_clust=False
        self.valid_prob=0.50
    
    def start_all(self):
        name="Start Clusters, Euclidian Distances, Cluster Resposability Matrix and Inverse of Euclideand Dist."
        self.main_header(name,"009")
        self.start_cluster_names() #1
        self.start_centers() #2
        self.start_complete_diff_array() #3
        self.start_complete_resp_array() #4
        self.start_inv_euclid_d_array() #5

    def start_cluster_names(self):
        name=("Starting Cluster Names: Will be used to track cluster status and properties")
        self.second_header(name,9,1)
        for clust in range(self.tt_clusters):
            if clust>10:
                name="Cluster - 0"+str(clust)
            else:
                name="Cluster - 00"+str(clust)
            self.clust_name_list.append(name)

    def start_centers(self):
        name=("Starting Cluster Center: Define Starting for each Cluster Center")
        self.second_header(name,9,2)
        self.clust_centers_M=np.zeros((self.tt_clusters,2))
        min_PCA0=self.pca_std_array[:,0].min()
        max_PCA0=self.pca_std_array[:,0].max()
        min_PCA1=self.pca_std_array[:,1].min()
        max_PCA1=self.pca_std_array[:,1].max()
        for clust in range(self.tt_clusters):
            if clust==0:
                self.clust_centers_M[clust]=np.array([max_PCA0,max_PCA1])
            elif clust==1:
                self.clust_centers_M[clust]=np.array([min_PCA0,min_PCA1])
            elif clust==2:
                self.clust_centers_M[clust]=np.array([min_PCA0,max_PCA1])
            elif clust==3:
                self.clust_centers_M[clust]=np.array([max_PCA0,min_PCA1])
            else:
                self.clust_centers_M[clust]=self.pca_std_array[np.random.choice(self.samples_number)]

    def start_complete_diff_array(self):
        name=("Starting Complete Diff Arrays: Calculate and Store Euclidean Distances of Clusters"+
        "and Other parameters such [xd,yd,thet,euclid dist,quad]")
        self.second_header(name,9,3)
        for clust in range(self.tt_clusters):
            if clust==0:
                self.complet_clust_diff_array=np.array([self.start_single_diff_array()])
            else:
                self.complet_clust_diff_array=np.vstack((
                    self.complet_clust_diff_array,
                    np.array([self.start_single_diff_array()])))

    def start_complete_resp_array(self):
        name=("Starting a Complete Array - Cluster Responsability : Shape => rows = samples, columns = clusters")
        self.second_header(name,9,4)
        for clust in range(self.tt_clusters):
            if clust==0:
                self.complet_clust_resp_array=self.start_single_resp_matrix()
            else:
                self.complet_clust_resp_array=np.hstack((
                    self.complet_clust_resp_array,
                    self.start_single_resp_matrix()))

    def start_inv_euclid_d_array(self):
        name=("Starting a Complete Array - Inverse of the Euclidean Distance: Shape => rows = samples, columns = clusters")
        self.second_header(name,9,5)
        for clust in range(self.tt_clusters):
            if clust==0:
                self.complet_inv_euclid_d_array=self.start_single_inv_euclid_d_matrix()
            else:
                self.complet_inv_euclid_d_array=np.hstack((
                    self.complet_inv_euclid_d_array,
                    self.start_single_inv_euclid_d_matrix()))
  
    def update_complete_diff_array(self):
        name=("Updating a Complete Array - Diff and Inverse")
        self.second_header(name,9,6)
        for clust in range(self.tt_clusters):
            for sample in range(self.samples_number):
                if self.complet_clust_resp_array[sample,clust]==1:
                    self.complet_clust_diff_array[clust,sample,:]=self.get_diff_series(
                        self.clust_centers_M[clust], self.pca_std_array[sample]
                        )
                    if self.complet_clust_diff_array[clust,sample,3]==0:
                        self.complet_clust_diff_array[clust,sample,3]=1/(10**30)
                    self.complet_inv_euclid_d_array[sample,clust]=(
                        1/self.complet_clust_diff_array[clust,sample,3]
                    )

    def update_clust_resp(self):
        name=("Updating a Complete Array - Responsability Matrix: "+
        "Update Responsabilities values for each Cluster")
        self.second_header(name,9,7)

        self.potential_resp=(self.complet_inv_euclid_d_array/
                            self.complet_inv_euclid_d_array.sum(axis=1, keepdims=True))

        for sample in range(self.potential_resp.shape[0]):
            self.complet_clust_resp_array[sample,:]=(
                self.potential_resp[sample,:]/
                self.potential_resp[sample,:].max()
            )
        self.get_points_in_cluster()

        #### div1 is similar to ttpoints_in_cluster but div1 will be modified to avoid division by 0
        div1=self.complet_clust_resp_mask.sum(axis=0, keepdims=True).T
        div1[div1==0]=1
        return div1

    def get_points_in_cluster(self):
        self.third_header("#Validate Respons. and count valid points in cluster",9,7,1)
        #### Mask1 - Ask: Are there respons. below the valid probability set initialy
        #### Answer: True=Yes, False=No
        mask1=self.complet_clust_resp_array[:,:]<self.valid_prob 
        #### Mask2 - Ask: Are there respons. equal or above the valid probability set initialy
        #### Answer: True=Yes, False=No
        self.complet_clust_resp_mask=np.logical_not(mask1)
        #### ** complet_clust_resp_mask and complet_clust_resp_array are almost the same except for:
        #### complet_clust_resp_mask is boolean array
        #### complet_clust_resp_array is float array

        #### Sum True Values of Mask2 - Return the number of points which probably belongs to the cluster
        #### This probability will be validated later
        self.ttpoints_in_cluster=self.complet_clust_resp_mask.sum(axis=0) #### Sum true values
        #### Update Responsability Matrix part 1 
        #### - All values below the threshold will be 0, otherwise it will 1       
        self.complet_clust_resp_array[mask1]=0
        self.complet_clust_resp_array[self.complet_clust_resp_mask]=1


    def update_centers(self):
        pass

    #class Cluster_resp_oper(Headers):
    # def __init(self):
    #     pass

    def cluster_resp_data(self,M):
        self.second_header("#Calculate Initial Responsabilities for each Cluster",9,4)

        self.quadrant_w_resp=(self.complet_clust_diff_array[:,:,4]*
              self.complet_clust_resp_array)
        ### Find in which quadrant is located most of the data. 
        self.count_quadrant()
        self.check_main_quadrants()

        ### Divide the maximum value for each cluster to find which the most important quadrant
        quad_main=self.quad_main_all[0]
        quad_sec=self.quad_main_all[1]
        
        ###### adjust all functions which requires quad_main and quad_sec

        # V2 and V3 DATA row=cluster, column=[theta_mean,theta_std,rd_mean,rd_std]
        V2=group_data_mean_and_std(M,self.quadrant_w_resp,
        quad_main,self.complet_clust_diff_array,self.complet_clust_resp_array)
         # V2 Main Quad
        V3=group_data_mean_and_std(M,self.quadrant_w_resp,
        quad_sec,self.complet_clust_diff_array,self.complet_clust_resp_array) # V3 Sec Quad
        V4=np.hstack((V2,V3))
        V5=np.zeros((V4.shape[0],1)) #### added to solve cluster_str_indx issue
        V4=np.hstack((V4,V5)) 

        main_map,main_map2=cluster_relation_map(self.complet_clust_diff_array,M,V4) #cluster_relation_map=>full_map,closest_map

        self.complet_clust_resp_array=cluster_resp_loop(main_map,M,quad_main,quad_sec,V4,self.complet_clust_diff_array,self.quadrant_sum,self.complet_clust_resp_array)
        self.complet_clust_resp_array=cluster_resp_loop(main_map2,M,quad_main,quad_sec,V4,self.complet_clust_diff_array,self.quadrant_sum,self.complet_clust_resp_array)

        mask=np.zeros((self.complet_clust_resp_array.shape))
        for sample in range(self.complet_clust_resp_array.shape[0]):
            mask[sample,:]=self.complet_clust_diff_array[sample,:,3]==self.complet_clust_diff_array[sample,:,3].min()
        
        for sample in range(self.complet_clust_resp_array.shape[0]):
            if self.complet_clust_resp_array[sample,:].sum()>1:
                self.complet_clust_resp_array[sample,:]=mask[sample,:].astype(int)


        return self.complet_clust_resp_array,V4

    def count_quadrant(self):
        self.third_header("#Count points per cluster quadrant",9,4,1)
        for q in range(4):
            qn=q+1
            if q==0:
                self.quadrant_sum=(self.quadrant_w_resp[:,:]==qn).sum(axis=0)
            else:
                self.quadrant_sum=np.vstack((
                    self.quadrant_sum,
                    (self.quadrant_w_resp[:,:]==qn).sum(axis=0)))

    def check_main_quadrants(self):
        self.third_header("Find Quadrant with most relevant data",9,4,2)
        quad_mult=np.array([[1],[2],[3],[4]])
        quad_sum_all=self.quadrant_sum
        for quad in range(4):
            quad_prio=quad_sum_all/quad_sum_all.max(axis=0)
            mask1=quad_prio==1
            quad_main=(quad_prio*quad_mult).sum(axis=0)
            if quad==0:
                self.quad_main_all=quad_main
            else:
                self.quad_main_all=np.vstack((self.quad_main_all,quad_main))

            quad_sum_all=quad_sum_all*np.logical_not(mask1)

if __name__ == '__main__':
    main()    
        


