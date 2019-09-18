import pandas as pd 

def main():
    file_path="C:/Users/ricar/Downloads/Experimento_TOF_Texas.xlsx"
    exp_dt=Exp_data(file_path)
    from IPython import embed; embed()

class Exp_data():
    def __init__(self,file_path):
        self.file_path=file_path
        self.df=pd.read_excel(self.file_path, header=0,index_col=0)
        print("Available columns")
        print(self.df.columns)
    
    def get_exp_data(self,exp_number,exp_column_name):
        if exp_number <=0:
            print("There is no exp_number with negative values or zero")
            print(self.df.head())
        else:
            data=self.df.loc[[exp_number],[exp_column_name]]
            value=data.values[0][0]
            return value
        
if __name__ == "__main__":
    main()
    pass


