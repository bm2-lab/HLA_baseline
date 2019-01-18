from pyspark import SparkConf, SparkContext
import subprocess

n_cpu = 38
conf = SparkConf().setAppName('abc').setMaster(f'local[{n_cpu}]')
sc = SparkContext.getOrCreate(conf=conf)


num_tr = 69
num_te = 5

dir_path = '/data/hla_data/train_test'
trlst = [f'train_sample_{i}' for i in range(num_tr)]
telst = [f'test_sample_{i}' for i in range(num_te)]

lst = trlst + telst

for i, fname in enumerate(lst):
    print(f'Processing {fname} ({i+1} / {len(lst)})...')
    rdd_line = sc.textFile(f'{dir_path}/{fname}.txt', minPartitions=n_cpu)
    pos_line = rdd_line.filter(lambda line: int(line.split('\t')[-1]) == 1)
    neg_line = rdd_line.filter(lambda line: int(line.split('\t')[-1]) == 0)
    pos_line.saveAsTextFile(f'{dir_path}/{fname}_pos_tmp')
    neg_line.saveAsTextFile(f'{dir_path}/{fname}_neg_tmp')
    pos_cmd = f'cat {dir_path}/{fname}_pos_tmp/* > {dir_path}/{fname}_pos.txt'
    neg_cmd = f'cat {dir_path}/{fname}_neg_tmp/* > {dir_path}/{fname}_neg.txt'
    subprocess.call(pos_cmd, executable='/bin/bash', shell=True)
    subprocess.call(neg_cmd, executable='/bin/bash', shell=True)


