from PIL import Image
import face_recognition
import os

#原始图片目录
Train_Data_Path = './data/Images/train_data/'
Test_Data_Path = './data/Images/test_data/'
#截取面部图片存放目录
Train_Face_Path = './data/Face/train_face/'
Test_Face_Path = './data/Face/test_face/'

#将图片中的面部截取出来并保存
def find_and_save_face(Original_Image_File,Face_Image_File):
    #加载.jpg格式的原始图片文件为numpy数组
    image = face_recognition.load_image_file(Original_Image_File)
    #找到原始图片中所有面部
    face_locations = face_recognition.face_locations(image)
    #打印出原始图片中发现的面部数
    print("在原始图片中发现了 {} 张面部 .".format(len(face_locations)))
    #图片中只有一张面部的情况
    if len(face_locations)==1:
        # 获取面部在原始图片中的位置
        top, right, bottom, left = face_locations[0]
        # 打印出面部在图片中的位置
        print("面部所在位置的各像素点位置 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # 截取面部图片
        face_image = image[top:bottom, left:right]
        # 将数组转化为图像
        pil_image = Image.fromarray(face_image)
        # 将面部图像保存下来
        pil_image.save(Face_Image_File)

    #原始图片中存在多张面部的情况，遍历面部列表
    else:
        i=0
        for face_location in face_locations:
            # 获取每一张面部在原始图片中的位置
            top, right, bottom, left = face_location
            # 打印出每一张面部在图片中的位置
            print("面部所在位置的各像素点位置 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            # 截取面部图片
            face_image = image[top:bottom, left:right]
            # 将数组转化为图像
            pil_image = Image.fromarray(face_image)
            Face_Image_File = Face_Image_File.split('.jpg')[0]+'-'+str(i)+'.jpg'
            # 将面部图像保存下来
            pil_image.save(Face_Image_File)
            i=i+1

def main():
    #获取目录下所有文件名
    train_list = os.listdir(Train_Data_Path)
    for image in train_list:
        Original_Image_File = Train_Data_Path+image
        Face_Image_File = Train_Face_Path+image
        try:
            find_and_save_face(Original_Image_File,Face_Image_File)
        except:
            print('error!')
    test_list = os.listdir(Test_Data_Path)
    for image in test_list:
        Original_Image_File = Test_Data_Path+image
        Face_Image_File = Test_Face_Path+image
        try:
            find_and_save_face(Original_Image_File,Face_Image_File)
        except:
            print('error!')

if '__name__==__main__':
    main()