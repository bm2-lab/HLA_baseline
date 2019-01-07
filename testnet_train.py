import tensorflow as tf
from examples.model.testnet import TestNet
import tfutil

model_dir = 'temp/testnet/mnist'
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
