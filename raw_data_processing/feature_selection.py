import sys
### Modify sys path if the tof project folder is not in PATH 
sys.path.append("D:\\tof")

#### Image and Array Operations Libraries
import numpy as np

#### Plot Libraries
import matplotlib.pyplot as plt

#### Machine Learning and Data Mining Libraries
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

#### Customized Libraries - Luhm
from experiment.std_headers import Headers
from bin_opener.input_data import Input_data

class Feature_selection(Input_data,Headers):
    def __init__(self,array):
        print("Table Array Shape",array.shape)
        self.main_header("Starting Data Standardization and Feature Selection","FS_001")
        self.raw_array=array

        self.std_array= StandardScaler().fit_transform(self.raw_array)
        self.titles = ["frame", "i_pxl", "j_pxl", 
                        "Amp",  "Amb", "Depth", "Dist",
                        "Phase",  "PC_x", "PC_y", "PC_z","PC_i"]

    # Step 2 - Get Covariance Matrix
    def cov_matrix(self,std_array):
        self.main_header("Standard Cov. Matrix","FS_002")
        self.cov_mat = np.cov(std_array.T)

    # Step 3 - Plot Cov Matrix as Heat Map 
    def plot_heatmap(self):
        self.main_header("Cov. Matrix Plot","FS_003")
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
        self.main_header("Find Relevant Cov Mat Features","FS_004")
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

        self.main_header("Revome Feature Columns Based on List","FS_005")
        self.std_array=np.delete(self.std_array,columns_list,1)
        self.cov_matrix(self.std_array)
        indexes = columns_list
        for index in sorted(indexes, reverse=True):
            del self.titles[index]

    # Step 6 Get EigenVector and EigenValues from Selected Features based on the Cov Matrix
    def get_eigen(self):
        self.main_header("Get EigenVector and EigenValues from Selected Features","FS_006")
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov_mat)
        self.eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:,i]) for i in range(len(self.eig_vals))]
        self.eig_pairs.sort()
        self.eig_pairs.reverse()

    # Step 7 Create Translation Matrix for Input Array to provide best PCA combinations
    def pca_list(self,plot_limit=1):
        self.main_header("Create Translation Matrix for Input Array to provide best PCA combinations","FS_007")
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
