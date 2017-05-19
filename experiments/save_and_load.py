import tensorflow as tf
import numpy as np
import sys

try:
    sys.path.insert(0, 'src')
    from Config import Config
except:
    pass

x = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]

xx = tf.Variable(x, name='xx')
m = tf.Variable(tf.random_normal(shape=[ 5]), name='m')
v = tf.Variable(tf.random_normal(shape=[ 5]), name='v')

init = tf.global_variables_initializer()
sess = tf.Session()

m_tensor, v_tensor = tf.nn.moments(xx,[0])
assign_op=m.assign(m_tensor)

print sess.run([init,m_tensor,assign_op])

# tf.add_to_collection('vars', xx)
# tf.add_to_collection('vars', m)
# tf.add_to_collection('vars', v)

saver = tf.train.Saver()
model_dir= 'tmp_tf/model/'
model_name='model'
if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)
tf.gfile.MakeDirs(model_dir)
saver.save(sess, model_dir+model_name)
print tf.global_variables()

print '-'*10

sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir+model_name+'.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_dir))

# all_vars = tf.get_collection('vars')
# print all_vars
#
# for v in all_vars:
#     v_ = sess.run(v)
#     print(v_)

print "res:"
# xx = tf.placeholder("float32", [5, 2],name='xx')
# m = tf.placeholder("float32", [1, 5], name='m')
# v = tf.placeholder("float32", [1, 5], name='v')

# init = tf.global_variables_initializer()
# graph = tf.get_default_graph()
# xx = graph.get_tensor_by_name('xx:0')
# m = graph.get_tensor_by_name('m:0')
# v = graph.get_tensor_by_name('v:0')

print tf.global_variables()
print sess.run('xx:0')
print 'mean is', sess.run('m:0')
# print sess.run('v:0')