import tensorflow as tf
import data
#模型保存路径
CKPT_MODEL_DIR = "./model/ckpt/"
CKPT_MODEL_NAME = "facerank"
PB_MODEL_DIR = "./model/pb/"
PB_FILE_NAME = "facerank.pb"

BATCH_SIZE = 100
def main():
    with tf.Session() as sess:
        # 检测保存的模型是否存在，如果存在则将模型加载到当前计算图中
        ckpt = tf.train.latest_checkpoint(CKPT_MODEL_DIR)
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
            accuracy = graph_def.get_tensor_by_name('accuracy:0')
            global_step = graph_def.get_tensor_by_name('global_step:0')
            tf.assign(global_step, start_step)

            graph = {'input': input, 'label': label, 'keep_prob': keep_prob, 'is_training': is_training,
                     'logits': logits, 'prediction': prediction, 'accuracy': accuracy,
                     'global_step': global_step}
            # 获取测试数据
            xs_test, ys_test = data.batch_data(isTrain=False, batch_size=BATCH_SIZE)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            while True:
                xs_test_batch, ys_test_batch = sess.run([xs_test, ys_test])
                # 测试喂入数据,测试时恢复所有全连接网络节点运算
                feed_test_dict = {graph['input']: xs_test_batch, graph['label']: ys_test_batch,
                                  graph['keep_prob']: 1.0,
                                  graph['is_training']: False}
                acc = sess.run(graph['accuracy'], feed_dict=feed_test_dict)
                global_step = sess.run(graph['global_step'])
                print('After %d training step(s),acc =%g' % (global_step, acc))
            coord.request_stop()
            coord.join(threads)
        else:
            print("No checkpoint file found!")


if __name__ == '__main__':
    main()
