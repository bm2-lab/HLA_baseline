from sklearn.externals import joblib
import pandas as pd

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

def hla_dict(file_name):
    index=1
    dict_hla={}
    with open(file_name) as f:
        for line in f:
            line=line.strip()
            dict_hla[line]=index
            index=index+1
    return dict_hla
import numpy as np

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


dict_hla = hla_dict('hla_all.csv')


df_sample_hla = pd.read_csv("sample_id_hla.csv")

dict_hla = pro_sample_hla_code(df_sample_hla, dict_hla)

joblib.dump((AA_dict, dict_hla) ,'hla_dict.pkl')

