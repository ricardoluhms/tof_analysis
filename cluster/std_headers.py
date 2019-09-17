
class Headers():
    def __init__(self):
        pass

    def main_header(self,name,number):
        print(""*100)
        print("-"*100)
        name="### "+str(number)+" - "+name
        print(name,"-"*(99-len(name)))
        print("-"*100)

    def second_header(self,name,number1,number2):
        name="### "+str(number1)+"-"+str(number2)+" - "+name
        print(name,"-"*(99-len(name)))

    def third_header(self,name,number1,number2,number3):
        name="### "+str(number1)+"-"+str(number2)+"-"+str(number3)+" - "+name
        print(name,"-"*(99-len(name)))
        
    def simple_header(self,name):
        name="###  - "+name
        print(name,"-"*(99-len(name)))
