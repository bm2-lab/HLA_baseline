from sklearn.externals import joblib
import tfutil
import numpy as np
from pyspark import SparkContext, SparkConf

n_cpu = 74
conf = SparkConf().setAppName('abc').setMaster(f'local[{n_cpu}]')
sc = SparkContext.getOrCreate(conf=conf)

dt_aa, dt_hla = joblib.load('hla_dict.pkl')


def peptide_code(peptide, AA_dict, maxlen=11):
    o = list(map(lambda x: AA_dict[x], peptide))
    k = len(o)
    o = o[:k // 2] + [0] * (maxlen - k) + o[k // 2:]
    return o


def generator_file(file_name, dict_sample_hla_code, AA_dict):
    sample_id = file_name.split('/')[-1][:-8]
    per_hla_code = dict_sample_hla_code[sample_id]
    gn = (i.strip().split('\t') for i in open(file_name) if i.strip() != '')
    gn = map(lambda x: [x[0], int(x[1])], gn)
    gn = map(lambda x: [peptide_code(x[0], AA_dict), x[1]], gn)
    gn = map(lambda x: [x[0] + per_hla_code, x[1]], gn)
    gn = map(lambda x: [np.array(x[0]), np.array(x[1])], gn)
    return gn


num_tr = 69
num_te = 5

dir_path = '/data/hla_data/raw'
trlst = [f'p10_data/train/train_sample_{i}' for i in range(num_tr)]
telst = [f'p10_data/test/test_sample_{i}' for i in range(num_te)]

lst = trlst + telst

save_path = '/data/hla_data'
rdd_names = sc.parallelize(lst, numSlices=n_cpu)

# num_examples_per_file = 80000000
def foo(fname):
    src_pos = f'{dir_path}/{fname}_pos.txt'
    src_neg = f'{dir_path}/{fname}_neg.txt'
    dst_pos = f'{save_path}/{fname}_pos'
    dst_neg = f'{save_path}/{fname}_neg'
    gn_pos = generator_file(src_pos, dt_hla, dt_aa)
    tfutil.write_tfrec_from_generator(gn_pos, dst_pos, 2)
    gn_neg = generator_file(src_neg, dt_hla, dt_aa)
    tfutil.write_tfrec_from_generator(gn_neg, dst_neg, 2)


rdd_names.foreach(foo)


