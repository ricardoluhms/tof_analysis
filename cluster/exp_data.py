import pandas as pd 
from std_headers import Headers

def main():
    #file_path="C:/Users/ricar/Downloads/Experimento_TOF_Texas.xlsx"
    #exp_dt=Exp_data(file_path)
    from IPython import embed; embed()
    #pass

class Exp_data(Headers):
    def __init__(self,file_path,exp_numb=0):
        self.main_header("Get Experiment Data Setup Parameters","EX_DT_001")
        self.file_path=file_path
        self.df=pd.read_excel(self.file_path, header=0,index_col=0)
        self.exp_numb=exp_numb
        self.simple_header("Available columns")
        print(self.df.columns)
        if exp_numb==0:
            self.simple_header("Experiment Number initially set as 0 but it must be changed for a value higher than 0")

    def get_exp_numb(self,exp_folder_path):
        self.exp_numb=int(exp_folder_path.split("Exp")[-1])
        
    def get_exp_data(self,exp_number,exp_column_name):
        if exp_number <=0:
            self.simple_header("There is no exp_number with negative values or zero")
            print(self.df.head())
        else:
            data=self.df.loc[[exp_number],[exp_column_name]]
            value=data.values[0][0]
            return value
        
if __name__ == "__main__":
    main()
    pass


