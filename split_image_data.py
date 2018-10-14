import os

Image_Path = './data/Images/'
Train_Label_File = './data/cross_validation_1/train_1.txt'
Test_Label_File = './data/cross_validation_1/test_1.txt'
Train_Data_Path = Image_Path+'/Train_Data/'
Test_Data_Path = Image_Path+'/Test_Data/'

def split(Train_Data_Path,Test_Data_Path):
    with open(Train_Label_File,'r') as train:
        for train_line in train:
            train_name = train_line.split(' ')[0]
            os.rename(Image_Path+train_name,Train_Data_Path+train_name)
    with open(Test_Label_File,'r') as test:
        for test_line in test:
            test_name = test_line.split(' ')[0]
            os.rename(Image_Path+test_name,Test_Data_Path+test_name)
def main():
    split(Train_Data_Path,Test_Data_Path)

if __name__ == '__main__':
    main()
