from sklearn.externals import joblib
import tfutil
import numpy as np
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('abc').setMaster('local[2]')
sc = SparkContext(conf=conf).getOrCreate()

dt_aa, dt_hla = joblib.load('hla_dict.pkl')


def peptide_code(peptide, AA_dict, maxlen=11):
    o = list(map(lambda x: AA_dict[x], peptide))
    k = len(o)
    o = o[:k // 2] + [0] * (maxlen - k) + o[k // 2:]
    return o


def generator_file(file_name, dict_sample_hla_code, AA_dict):
    sample_id = file_name.split('.')[0]
    per_hla_code = dict_sample_hla_code[sample_id]
    gn = (i.strip().split('\t') for i in open(file_name) if i.strip() != '')
    gn = map(lambda x: [x[0], int(x[1])], gn)
    gn = map(lambda x: [peptide_code(x[0], AA_dict), x[1]], gn)
    gn = map(lambda x: [x[0] + per_hla_code, x[1]], gn)
    gn = map(lambda x: [np.array(x[0]), np.array(x[1])], gn)
    return gn


fnames = ['train_sample_0', 'train_sample_3']
rdd_names = sc.parallelize(fnames)


def foo(fname):
    gn = generator_file(f'{fname}.txt', dt_hla, dt_aa)
    tfutil.write_tfrec_from_generator(gn, fname, 2)


rdd_names.foreach(foo)

# gn = generator_file(f'{fname}.txt', dt_hla, dt_aa)
# tfutil.write_tfrec_from_generator(gn, fname, 2)
