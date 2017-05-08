from __future__ import print_function

import tensorflow as tf
import os
import os.path as osp
import subprocess
os.getenv("KERAS_BACKEND", "tensorflow")


class Config(object):
    # Parameters
    dbg=True
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 1 if dbg else 128
    # batch_size = 1
    display_step = 10

    source_dim = 8192
    hidden1 = 2048
    hidden2 = 512
    hidden3 = 128
    start_learning_rate = 0.001
    decay_rate = 0.96
    decay_step = 2000
    counts = 10000
    max_epoch = 20
    # Network Parameters
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

    def __init__(self):
        self.root_dir="/home/xlwang/NetworkCompress/"
        self.log_dir = self.root_dir+"tmp_tf/log/"
        self.save_dir = self.root_dir+"tmp_tf/save/"
        self.data_dir= self.root_dir + "tmp_tf/data/"
        self.tmp_dit=self.root_dir+"tmp_tf/"
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
        )
        self.sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    def check_path(self):
        if tf.gfile.Exists(config.log_dir):
            tf.gfile.DeleteRecursively(config.log_dir)

            cmd="rm -rf "+config.log_dir
            subprocess.call(cmd.split())
            assert not osp.exists(config.log_dir), config.log_dir+" still exits..."
        tf.gfile.MakeDirs(config.log_dir)

        if not tf.gfile.Exists(config.save_dir):
            tf.gfile.MakeDirs(config.save_dir)

        if not tf.gfile.Exists(config.data_dir):
            tf.gfile.MakeDirs(config.data_dir)

config = Config()
config.check_path()

def variable_summaries(var, name):
    with tf.name_scope(name=name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias",b)
        tf.summary.histogram("activation",act)
        # act_list=tf.split(act,size_out,axis=)
        print(act.get_shape())
        # tf.Print(act,[act],message="!!!!!")
        # tf.Print(act,[act.get_shape()],message="!!!")
        # tf.Print(act,[tf.shape(act)],message="!!!!")

        x_min = tf.reduce_min(w)
        x_max = tf.reduce_max(w)
        weights_0_to_1 = (w - x_min) / (x_max - x_min)
        weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)

        # to tf.image_summary format [batch_size, height, width, channels]
        weights_transposed = tf.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])
        tf.summary.image('activation',weights_transposed)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Add fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name)as scope:
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("bias", b)
        tf.summary.histogram("activation", act)
        return act


def mnist_model():
    with tf.device("/cpu:0"):
        tf.reset_default_graph()
        sess=tf.Session(config=config.sess_config)

        x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)
        y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

        conv1=conv_layer(x_image,1,64,"conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc")

        with tf.name_scope("xent"):
            xent = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=y), name="xent")
            tf.summary.scalar("xent", xent)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(0.001).minimize(xent)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        summaries = tf.summary.merge_all()

        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer=tf.summary.FileWriter(config.log_dir)
        writer.add_graph(sess.graph)

        for i in range(2001):
            batch = mnist.train.next_batch(config.batch_size)
            if i % 1 == 0:
                [train_accuracy, s] = sess.run([accuracy, summaries], feed_dict={x: batch[0], y: batch[1]})
                writer.add_summary(s, i)

            if i % 500 == 0:
            #     sess.run(accuracy, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
                saver.save(sess, config.save_dir+ "mnist",global_step=i)
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(config.data_dir, one_hot=True)

def main():

    mnist_model()

if __name__=='__main__':
    main()