import os.path
import shutil
import tensorflow as tf
import data
from tensorflow.contrib import slim


CKPT_MODEL_DIR = "drive/FaceRank/model/ckpt/"
CKPT_MODEL_NAME = "facerank"
PB_MODEL_DIR = "drive/FaceRank/model/pb/"
PB_FILE_NAME = "facerank.pb"
BATCH_SIZE = 100
STEPS=5001

def build_graph(input_size, n_class):
    #定义输入的数据和标签，全连接层输出保存的概率以及是否为训练的标志位
    input = tf.placeholder(tf.float32, [None, input_size, input_size, 3], name='x-input')
    label = tf.placeholder(tf.float32, [None, n_class], name='y-input')
    keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
    is_training = tf.placeholder(tf.bool, [], name='train_flag')
    #进行卷积操作
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            kernel_size=[3, 3]):
            conv1 = slim.conv2d(input, 32, scope='conv1')
            pool1 = slim.max_pool2d(conv1, kernel_size=[2, 2], scope='pool1')
            conv2 = slim.conv2d(pool1, 64, scope='conv2')

        flatten = slim.flatten(conv2)
        fully1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, slim.nn.relu, scope='fully1')
        #逻辑回归后每一个类别对应一个概率
        logits = slim.fully_connected(slim.dropout(fully1, keep_prob), n_class, None, scope='logits')
        #预测结果
        prediction = tf.argmax(logits, 1, name='prediction')
    #损失函数
    loss = tf.reduce_mean(slim.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='loss'))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), tf.float32), name='accuracy')
    #训练的总轮数
    global_step = tf.Variable(0, trainable=False, name='global_step')
    #优化器定义
    optimizer = slim.train.AdamOptimizer(1e-4)
    #参数更新操作绑定
    updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updata_ops):
        train_op = slim.learning.create_train_op(loss, optimizer, global_step)

    return {'input': input,
            'label': label,
            'keep_prob': keep_prob,
            'is_training': is_training,
            'logits': logits,
            'prediction': prediction,
            'accuracy': accuracy,
            'loss': loss,
            'train_op': train_op,
            'global_step': global_step}


def main():
    if not os.path.exists(CKPT_MODEL_DIR):  # 创建目录
        os.mkdir(CKPT_MODEL_DIR)
    if os.path.exists(PB_MODEL_DIR):  # 删除目录
        shutil.rmtree(PB_MODEL_DIR)
    if not os.path.exists(PB_MODEL_DIR):  # 创建目录
        os.mkdir(PB_MODEL_DIR)


    with tf.Session() as sess:
        #检测模型是否存在
        ckpt = tf.train.latest_checkpoint(CKPT_MODEL_DIR)
        #存在则断点续训
        if ckpt:
            saver = tf.train.import_meta_graph(ckpt + '.meta')
            saver.restore(sess, ckpt)
            graph_def = tf.get_default_graph()
            start_step = int(ckpt.split('-')[-1])

            input = graph_def.get_tensor_by_name('x-input:0')
            label = graph_def.get_tensor_by_name('y-input:0')
            keep_prob = graph_def.get_tensor_by_name('keep_prob:0')
            is_training = graph_def.get_tensor_by_name('train_flag:0')
            logits = graph_def.get_tensor_by_name('logits/BatchNorm/Reshape_1:0')
            prediction = graph_def.get_tensor_by_name('prediction:0')
            loss = graph_def.get_tensor_by_name('Mean:0')
            accuracy = graph_def.get_tensor_by_name('accuracy:0')
            train_op = graph_def.get_tensor_by_name('train_op/control_dependency:0')
            global_step = graph_def.get_tensor_by_name('global_step:0')
            tf.assign(global_step, start_step)

            graph = {'input': input, 'label': label, 'keep_prob': keep_prob, 'is_training': is_training,
                     'logits': logits, 'prediction': prediction, 'accuracy': accuracy, 'loss': loss,
                     'train_op': train_op, 'global_step': global_step}

            xs_train, ys_train = data.batch_data(isTrain=True,batch_size=BATCH_SIZE)
            #xs_test, ys_test = get_data.batch_data(isTrain=False,batch_size=BATCH_SIZE)
            # 创建喂入数据的队列
            coord = tf.train.Coordinator()
            # 开启线程
            threads = tf.train.start_queue_runners(sess, coord)
            for i in range(STEPS):
                xs_train_batch, ys_train_batch = sess.run([xs_train, ys_train])
                #xs_test_batch, ys_test_batch = sess.run([xs_test, ys_test])
                # 训练喂入数据，训练时，每次前向传播的过程中随机将部分节点输出改为0，防止过拟合提高模型稳定性
                feed_train_dict = {graph['input']: xs_train_batch, graph['label']: ys_train_batch,
                                   graph['keep_prob']: 0.5,
                                   graph['is_training']: True}
                # 测试喂入数据,测试时恢复所有全连接网络节点运算
                '''feed_test_dict = {graph['input']: xs_test_batch, graph['label']: ys_test_batch,
                                  graph['keep_prob']: 1.0,
                                  graph['is_training']: False}'''
                # 开启训练
                sess.run(graph['train_op'], feed_dict=feed_train_dict)
                loss_value = sess.run(graph['loss'],feed_dict=feed_train_dict)
                step = sess.run(graph['global_step'])
                if i % 50 == 0 and i != 0:
                    print('After training %d step(s),loss is %g'%(step,loss_value))
                    #print("test_accuracy == ", sess.run(graph['accuracy'], feed_dict=feed_test_dict))
                # 每训练1000轮将模型保存一次
                if i % 1000 == 0 and i != 0:
                    saver = tf.train.Saver()  # 声明saver用于保存模型
                    saver.save(sess, os.path.join(CKPT_MODEL_DIR, CKPT_MODEL_NAME),
                               global_step=graph['global_step'])  # 模型保存

                    # 将模型中的变量和参数固化成常量，以及保存模型调用时需要使用的节点名
                    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, sess.graph_def, ['x-input', 'keep_prob', 'train_flag', 'prediction'])

                    # 保存图为pb文件
                    with open(PB_MODEL_DIR + PB_FILE_NAME, 'wb') as f:
                        f.write(frozen_graph_def.SerializeToString())
            coord.request_stop()
            coord.join(threads)
        else:
            #创建计算图，64为矩阵大小，10为分类数
            graph = build_graph(64, 10)
            sess.run(tf.global_variables_initializer())
            xs_train,ys_train = data.batch_data(isTrain=True,batch_size=BATCH_SIZE)
            #xs_test,ys_test = get_data.batch_data(isTrain=False,batch_size=BATCH_SIZE)
            #创建喂入数据的队列
            coord = tf.train.Coordinator()
            #开启线程
            threads = tf.train.start_queue_runners(sess, coord)
            for i in range(STEPS):
                xs_train_batch,ys_train_batch=sess.run([xs_train,ys_train])
                #xs_test_batch,ys_test_batch=sess.run([xs_test,ys_test])
                # 训练喂入数据，训练时，每次前向传播的过程中随机将部分节点输出改为0，防止过拟合提高模型稳定性
                feed_train_dict = {graph['input']: xs_train_batch, graph['label']: ys_train_batch,
                                   graph['keep_prob']: 0.5,
                                   graph['is_training']: True}
                # 测试喂入数据,测试时恢复所有全连接网络节点运算
                '''feed_test_dict = {graph['input']: xs_test_batch, graph['label']: ys_test_batch,
                                  graph['keep_prob']: 1.0,
                                  graph['is_training']: False}'''
                # 开启训练
                sess.run(graph['train_op'], feed_dict=feed_train_dict)
                loss_value = sess.run(graph['loss'],feed_dict=feed_train_dict)
                step = sess.run(graph['global_step'])
                if i % 50 == 0 and i != 0:
                    print('After training %d step(s),loss is %g' % (step, loss_value))
                    #print("test_accuracy == ", sess.run(graph['accuracy'], feed_dict=feed_test_dict))
                #每训练1000轮将模型保存一次
                if i % 1000 == 0 and i != 0:
                   saver = tf.train.Saver()  # 声明saver用于保存模型
                   saver.save(sess, os.path.join(CKPT_MODEL_DIR, CKPT_MODEL_NAME),
                           global_step=graph['global_step'])  # 模型保存

                    #将模型中的变量和参数固化成常量，以及保存模型调用时需要使用的节点名
                   frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                     sess, sess.graph_def, ['x-input', 'keep_prob', 'train_flag', 'prediction'])

                   #保存图为pb文件
                   with open(PB_MODEL_DIR+PB_FILE_NAME, 'wb') as f:
                       f.write(frozen_graph_def.SerializeToString())
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()
