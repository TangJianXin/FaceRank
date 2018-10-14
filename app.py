import tensorflow as tf
import face_recognition
import numpy as np
from PIL import Image

#模型保存路径
CKPT_MODEL_DIR = "./model/ckpt/"
CKPT_MODEL_NAME = "facerank"

#将图片转化成矩阵
def get_data(ImageFile):
    input = []
    #加载图片
    image = face_recognition.load_image_file(ImageFile)
    #获得图片中的面部集合
    face_locations = face_recognition.face_locations(image)
    #打印出原始图片中发现的面部数
    print("在原始图片中发现了 {} 张面部 .".format(len(face_locations)))
    # 图片中只有一张面部的情况
    if len(face_locations) == 1:
        # 获取面部在原始图片中的位置
        top, right, bottom, left = face_locations[0]
        # 打印出面部在图片中的位置
        print("面部所在位置的各像素点位置 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # 截取面部图片
        face_image = image[top:bottom, left:right]
        # 将数组转化为图像
        pil_image = Image.fromarray(face_image)
        #重新设置图片大小
        resize_image = pil_image.resize((64,64))
        #将图片化为像素矩阵
        face = np.asarray(resize_image, dtype='float32')
        face = np.reshape(face, [64, 64, 3])
        input.append(face)
        data = np.array(input)
        data = np.reshape(data, [-1, 64, 64, 3])
        return data

    # 原始图片中存在多张面部的情况，遍历面部列表
    else:
        i = 0
        for face_location in face_locations:
            # 获取每一张面部在原始图片中的位置
            top, right, bottom, left = face_location
            # 打印出每一张面部在图片中的位置
            print("面部所在位置的各像素点位置 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            # 截取面部图片
            face_image = image[top:bottom, left:right]
            # 将数组转化为图像
            pil_image = Image.fromarray(face_image)
            # 重新设置图片大小
            resize_image = pil_image.resize((64, 64))
            # 将图片化为像素矩阵
            face = np.asarray(resize_image, dtype='float32')
            face = np.reshape(face, [64, 64, 3])
            input.append(face)
            i = i + 1
        data = np.array(input)
        data = np.reshape(data, [-1, 64, 64, 3])
        return data

def predict(ImagePath):
    with tf.Session() as sess:
        # 检测保存的模型是否存在，如果存在则将模型加载到当前计算图中
        ckpt = tf.train.latest_checkpoint(CKPT_MODEL_DIR)
        if ckpt:
            saver = tf.train.import_meta_graph(ckpt + '.meta')
            saver.restore(sess, ckpt)
            graph_def = tf.get_default_graph()

            input = graph_def.get_tensor_by_name('x-input:0')
            keep_prob = graph_def.get_tensor_by_name('keep_prob:0')
            is_training = graph_def.get_tensor_by_name('train_flag:0')
            prediction = graph_def.get_tensor_by_name('prediction:0')

            graph = {'input': input,'keep_prob': keep_prob, 'is_training': is_training,
                     'prediction': prediction,}
            try:
                input_data = get_data(ImagePath)
                num = input_data.shape[0]
                input_dict = {graph['input']: input_data, graph['keep_prob']: 1.0,
                                  graph['is_training']: False}
                if num == 0:
                    return '不能识别！'
                else:
                    predict = sess.run(graph['prediction'], feed_dict=input_dict)
                    print('这张脸的颜值为：',(predict[0]))
                    return '你的颜值为:'+' '+str(predict[0])+'(满分10分)'
            except:
                return '不能识别！'
        else:
            print("No checkpoint file found!")




