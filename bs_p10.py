import tensorflow as tf
import sonnet as snt
import tfutil
from tensorflow.contrib import slim
import numpy as np


class BaseModel(snt.AbstractModule):
    def __init__(self, keep_prob=0.75, nAA=20, aa_per_sample=11, nHLA=100, hla_per_sample=6, name='BaseModel'):
        super(BaseModel, self).__init__(name=name)
        self.keep_prob = keep_prob
        self.nAA = nAA
        self.aa_per_sample = aa_per_sample
        self.nHLA = nHLA
        self.hla_per_sample = hla_per_sample
        self.embedding_mat = np.eye(self.nHLA + 1, dtype=np.int32)[:, 1:]

    def _build(self, inputs, is_training):
        pep_x, hla_x = tf.split(tf.cast(inputs, dtype=tf.int32),
                                num_or_size_splits=[self.aa_per_sample, self.hla_per_sample], axis=-1)
        pep_x = tf.one_hot(pep_x, depth=self.nAA + 1)
        pep_net = snt.BatchFlatten()(pep_x)
        pep_net = snt.Linear(256)(pep_net)
        pep_net = tf.nn.relu(pep_net)
        pep_net = slim.dropout(pep_net, self.keep_prob, is_training=is_training)
        pep_logits = snt.Linear(self.nHLA)(pep_net)

        hla_net = tf.nn.embedding_lookup(self.embedding_mat, hla_x)
        hla_net = tf.reduce_sum(hla_net, axis=1)
        logits = tf.reduce_sum(tf.nn.sigmoid(pep_logits) * tf.cast(hla_net, dtype=tf.float32), axis=-1)
        return logits


def loss_f(labels, logits):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=logits))


import os

data_dir = '/data/hla_data/p10_data'
train_all = [k for k in os.listdir(f'{data_dir}/train') if k.endswith('tfrec')]
num_tr = 69
assert len(train_all) == num_tr * 2, f'Number of training data must be {num_tr * 2}'
train_pos = [f'{data_dir}/train/{k}' for k in train_all if '_pos.' in k]
train_neg = [f'{data_dir}/train/{k}' for k in train_all if '_neg.' in k]
assert len(train_pos) == num_tr, f'Number of positive training data must be {num_tr}'
assert len(train_neg) == num_tr, f'Number of negative training data must be {num_tr}'

batch_size_per_class = 1000
train_input_func = tfutil.balanced_read_tfrec_func([train_pos, train_neg], batch_size_per_class)

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 17])
labels = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)
net = BaseModel()

loss_func = tfutil.LossFunc(loss_f)
optimizer = tf.train.AdamOptimizer()

model_dir = 'temp/bs_p10'

n_gpu = 4
model_tensors = tfutil.ModelTensors(inputs, labels, is_training, net, train_input_func, loss_func, optimizer)
model = tfutil.TFModel(model_tensors, n_gpu, model_dir, training=True)

auc_op = tfutil.metrics_auc(labels, model.logits, curve='ROC', name='roc_auc')
pr_op = tfutil.metrics_auc(labels, model.logits, curve='PR', summation_method='careful_interpolation', name='pr_auc')
metric_opdefs = [auc_op, pr_op]

test_all = [k for k in os.listdir(f'{data_dir}/test') if k.endswith('tfrec')]
num_te = 5
listeners = []
for i in range(num_te):
    ts = sorted([f'{data_dir}/test/{k}' for k in test_all if f'test_sample_{i}_' in k])
    gnte = tfutil.read_tfrec(ts, batch_size_per_class * 3, shuffle=False)
    listener = tfutil.Listener(f'ts{i}', gnte, metric_opdefs)
    listeners.append(listener)

num_steps = 20000000
summ_steps = 10000
ckpt_steps = 100000
model.train(num_steps, summ_steps, ckpt_steps,
            metric_opdefs, None,
            listeners, max_ckpt_to_keep=10, from_scratch=True)
