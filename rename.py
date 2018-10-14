import os

Train_Label_File = './data/cross_validation_1/train_1.txt'
Test_Label_File = './data/cross_validation_1/test_1.txt'
Train_Data_Path = './data/Images/Train_Data/'
Test_Data_Path = './data/Images/Test_Data/'
#将图片重命名，以1-500的图片编号和对应的分数命名，分数为1-10分；例如：1-10.jpg
def rename():
    #遍历文件重命名图片
    with open(Train_Label_File,'r') as train:
        for train_line in train:
            train_name = train_line.split(' ')[0]
            train_score = train_line.split(' ')[1]
            train_score = float(train_score)
            #将原来的分数*2，原来为1-5分
            train_score = str(round(train_score*2))
            original_name = Train_Data_Path+train_name
            new_name = Train_Data_Path+train_score+'-'+train_name
            os.rename(original_name,new_name)
    with open(Test_Label_File,'r') as test:
        for test_line in test:
            test_name = test_line.split(' ')[0]
            test_score = test_line.split(' ')[1]
            test_score = float(test_score)
            test_score = str(round(test_score*2))
            original_name = Test_Data_Path+test_name
            new_name = Test_Data_Path+test_score+'-'+test_name
            os.rename(original_name,new_name)
def main():
    rename()
if __name__=='__main__':
    main()
