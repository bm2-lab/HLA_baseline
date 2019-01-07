import tensorflow as tf
import sonnet as snt
from tensorflow.contrib import slim
import tfutil

class TestNet(snt.AbstractModule):
    def __init__(self,
                 num_classes,
                 dropout_keep_prob=0.5,
                 name='testnet'):
        super(TestNet, self).__init__(name=name)
        self._num_classes = num_classes
        self._dropout_keep_prob = dropout_keep_prob

    @classmethod
    def arg_score(cls, is_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d], padding='SAME'):
                    with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                        return arg_sc

    def _build(self, inputs, is_training):
        with slim.arg_scope(self.arg_score(is_training)):
            net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='maxpool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='maxpool2')
            net = snt.BatchFlatten()(net)
            net = slim.fully_connected(net, 1024, scope='fc1')
            net = slim.dropout(net, keep_prob=self._dropout_keep_prob, scope='dp1')
            net = slim.fully_connected(net, self._num_classes, activation_fn=None,
                                       scope='fc2')
            return net


model_dir = 'temp/testnet'
batch_size = 100

data_dir = '/data/testdata/mnist'
data_tr = f'{data_dir}/mnist_tr.55000_28_28_1.10.tfrec'
data_tv = f'{data_dir}/mnist_tv.5000_28_28_1.10.tfrec'
data_te = f'{data_dir}/mnist_te.10000_28_28_1.10.tfrec'

train_input_func = tfutil.read_tfrec_func(data_tr, batch_size=batch_size)

num_classes = 10
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
labels = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)

network = TestNet(num_classes, dropout_keep_prob=0.5)

loss_func = tfutil.LossFunc(tfutil.create_loss, scope='loss')
learning_rate_init = 1e-3
learning_rate_decay_steps = 500
learning_rate = tfutil.dynamic_learning_rate(learning_rate_init, learning_rate_decay_steps)
optimizer = tf.train.AdamOptimizer(learning_rate)

summ_learning_rate = tf.summary.scalar('train/learning_rate', learning_rate)
summ_img = tf.summary.image('image', inputs, max_outputs=6)
extra_summ_ops = [summ_learning_rate, summ_img]

n_gpu = 2
model_tensors = tfutil.ModelTensors(inputs, labels, is_training, network, train_input_func, loss_func, optimizer)
model = tfutil.TFModel(model_tensors, n_gpu, model_dir, training=True)

acc_op = tfutil.metrics_accuracy(labels, model.logits)
mpca_op = tfutil.metrics_mean_per_class_accuracy(labels, model.logits, num_classes)
metric_opdefs = [acc_op, mpca_op]

gntv = tfutil.read_tfrec(data_tv, batch_size * 2, shuffle=False)
gnte = tfutil.read_tfrec(data_te, batch_size * 2, shuffle=False)

valid_listener = tfutil.Listener('validation', gntv, [acc_op])
test_listener = tfutil.Listener('test', gnte, metric_opdefs)
listeners = [valid_listener, test_listener]

num_steps = 600
summ_steps = 10
ckpt_steps = 100
model.train(num_steps, summ_steps, ckpt_steps,
            metric_opdefs, extra_summ_ops,
            listeners, max_ckpt_to_keep=5, from_scratch=True)
