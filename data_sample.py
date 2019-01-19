from pyspark import SparkConf, SparkContext
import subprocess

n_cpu = 38
conf = SparkConf().setAppName('abc').setMaster(f'local[{n_cpu}]')
sc = SparkContext.getOrCreate(conf=conf)



num_tr = 69
num_te = 5

ori_path = '/data/hla_data/raw/full_data_tpm'
dst_path = '/data/hla_data/raw/p10_data_tpm'
tr_neg = [f'train/train_sample_{i}_neg' for i in range(num_tr)]
te_neg = [f'test/test_sample_{i}_neg' for i in range(num_te)]

lst = tr_neg + te_neg

for i, fname in enumerate(lst):
    print(f'Processing {fname} ({i+1} / {len(lst)})...')
    rdd_ori = sc.textFile(f'{ori_path}/{fname}.txt', minPartitions=n_cpu)
    rdd_sam = rdd_ori.sample(False, 0.1)
    rdd_sam.saveAsTextFile(f'{dst_path}/{fname}_tmp')
    cmd = f'cat {dst_path}/{fname}_tmp/* > {dst_path}/{fname}.txt'
    subprocess.call(cmd, executable='/bin/bash', shell=True)


