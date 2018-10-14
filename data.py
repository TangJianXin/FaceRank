import tensorflow as tf
import numpy as np
import os
from PIL import Image

Train_Data_Path = './data/ResizeFace/train_data/'
Test_Data_Path = './data/ResizeFace/test_data/'

#读取图片数据
def read_data(isTrain):
    data = []
    label = []
    if isTrain:
        images = os.listdir(Train_Data_Path)
        Path = Train_Data_Path
    else:
        images = os.listdir(Test_Data_Path)
        Path = Test_Data_Path
    for image in images:
        score_tag = image.find('-')
        score = image[0:score_tag]
        img = Image.open(Path+image)
        x = np.asarray(img, dtype='float32')
        x = np.reshape(x, [-1, 64, 64, 3])
        data.append(x)
        y = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y[int(score) - 1] = 1
        y = np.reshape(y, [10, ])
        label.append(y)
    data = np.array(data)
    data = np.reshape(data,[-1,64,64,3])
    label = np.array(label)
    label = np.reshape(label,[-1,10])
    return data,label
def batch_data(isTrain,batch_size):
    x,y = read_data(isTrain=isTrain)
    #通过TensorFlow提供的管道队列形式循环读取训练数据
    input_queue = tf.train.slice_input_producer([x,y],shuffle=False)
    #将队列中的数据打乱读取
    data_batch,label_batch = tf.train.shuffle_batch(input_queue,batch_size=batch_size,num_threads=2,capacity=200,min_after_dequeue=60)
    return data_batch, label_batch