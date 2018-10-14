from PIL import Image
import os

Train_Face_Path = './data/Face/train_face/'
Test_Face_Path = './data/Face/test_face/'
Resize_train_Path = './data/ResizeFace/train_data/'
Resize_test_Path = './data/ResizeFace/test_data/'

#重新设置图片尺寸
def resize_image(ImageFile,SaveFile):

    img = Image.open(ImageFile)
    resize_image = img.resize((64,64))
    resize_image.save(SaveFile)

def main():
    train_images = os.listdir(Train_Face_Path)
    for image in train_images:
        ImageFile = Train_Face_Path+image
        SaveFile = Resize_train_Path+image
        resize_image(ImageFile,SaveFile)
    test_images = os.listdir(Test_Face_Path)
    for image in test_images:
        ImageFile = Test_Face_Path+image
        SaveFile = Resize_test_Path+image
        resize_image(ImageFile,SaveFile)

if __name__=='__main__':
    main()