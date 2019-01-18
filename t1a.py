import numpy as np
import pandas as pd
def hla_dict(file_name):
    index=1
    dict_hla={}
    with open(file_name) as f:
        for line in f:
            line=line.strip()
            dict_hla[line]=index
            index=index+1
    return dict_hla


def pro_sample_hla_code(df_sample_hla,dict_hla):
    dict_sample_hla_code={}
    sample_hla=np.array(df_sample_hla).tolist()
    for i in range(len(sample_hla)):
        sample_id=sample_hla[i][0]
        per_hla=sample_hla[i][1:]
        per_hla_code=[]
        for hla in per_hla:
            per_hla_code.append(dict_hla[hla])
            per_hla_code.sort()
            dict_sample_hla_code[sample_id]=per_hla_code


    return dict_sample_hla_code

def peptide_code(peptide,AA_dict,maxlen=11):
    o = list(map(lambda x: AA_dict[x], peptide))
    k = len(o)
    o = o[:k // 2] + [0] * (maxlen - k) + o[k // 2:]
    return o



def generator_file(file_name,dict_sample_hla_code,AA_dict):

    sample_id = file_name.split('.')[0]
    per_hla_code = dict_sample_hla_code[sample_id]
    gn=(i.strip().split('\t') for i in open("train_test/"+file_name) if i.strip() != "")
    gn = map(lambda x: [x[0],float(x[1]), int(x[2])], gn)
    gn=map(lambda x: [peptide_code(x[0],AA_dict),x[1],x[2]],gn)
    gn=map(lambda x: [x[0]+per_hla_code+[x[1]],x[2]],gn)
    gn=map(lambda x:[np.array(x[0]),np.array(x[1])],gn)
    return gn
if __name__=="__main__":
    AA_dict = {'A': 1,
               'C': 2,
               'D': 3,
               'E': 4,
               'F': 5,
               'G': 6,
               'H': 7,
               'I': 8,
               'K': 9,
               'L': 10,
               'M': 11,
               'N': 12,
               'P': 13,
               'Q': 14,
               'R': 15,
               'S': 16,
               'T': 17,
               'V': 18,
               'W': 19,
               'Y': 20,
               'O': 21,
               'U': 22,
               'Z': 0}
    dict_hla = hla_dict("hla_all.csv")
    df_sample_hla = pd.read_csv("sample_id_hla.csv")
    dict_sample_hla_code = pro_sample_hla_code(df_sample_hla, dict_hla)
    gn=generator_file("test_sample_1.txt", dict_sample_hla_code, AA_dict)