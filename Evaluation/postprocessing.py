import re, nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import file as ufile
import pandas as pd

data = ufile.read_csv("/Users/luyu/Downloads/ValX_demo/diabetes_criteria_test_20words_3lab20words.csv")

# extract lab-value statement for 3 lab tests
q_id = [d[0] for d in data]
q_inc = [d[1] for d in data]
q_ques = [d[2] for d in data]
q_fom = [d[3] for d in data]
glu_exp = [d[4] for d in data]
a1c_exp = [d[5] for d in data]
cre_exp = [d[6] for d in data]

stop = "gm|%,|hr|hrs|min|mins|minute|minutes|hour|hours|okay hour|day|days|week|weeks|month|months|yr|yrs|year|years".split("|")
#for d in glu_exp:print(d)

glu_exp_n = []
cre_exp_n = []

for c in glu_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
    ch = ch.replace('=,', '=')
    ch = ch.replace('absolute Glucose', 'absolute_Glucose')
    ch = ch.replace('average Glucose', 'average_Glucose')
    ch = ch.replace('mean Glucose', 'mean_Glucose')
    ch = ch.replace('average glucose', 'average_glucose')
    ch = ch.replace('Glucose tolerance', 'Glucose_tolerance')
    ch = ch.replace('serum glucose level', 'serum_glucose_level')
    ch = ch.replace('plasma glucose result', 'plasma_glucose_result')
    ch = ch.replace('blood glucose measurement', 'blood_glucose_measurement')
    ch = ch.replace('elevates my glucose-level', 'elevates_my_glucose_level')
    ch = re.sub('(?<=\w)(\,)', '', ch)
    ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
    seg = ch.split()
    if len(seg) > 3 and seg[3] in stop:
        del seg[:4]
    if len(seg) > 3 and seg[3] in stop:
        del seg[:4]
    if 0 < len(seg) < 3:
        del seg[:]
    if len(seg) > 3 and 'mg/ml' in seg[3]:
        seg[2] = str(int(seg[2])/100)
    if len(seg) > 2 and seg[2] in stop:
        del seg[:3]
    if len(seg) > 3 and seg[3] in stop:
        del seg[:4]
    #print(seg)
    glu_exp_n.append(seg)

del glu_exp_n[70][:3]
del glu_exp_n[762][:]
del glu_exp_n[1018][:]
del glu_exp_n[1060][:]
del glu_exp_n[2132][:]
del glu_exp_n[2849][:]
del glu_exp_n[3002][:3]
del glu_exp_n[3155][:]
del glu_exp_n[3156][:]
#for g in glu_exp_n:print(g)

for c in cre_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
    ch = ch.replace('=,', '=')
    ch = ch.replace('Creatinine ratio', 'Creatinine_ratio')
    ch = ch.replace('serum creatinine level', 'serum_creatinine_level')
    ch = re.sub('(?<=\w)(\,)', '', ch)
    ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
    seg = ch.split()
    if len(seg) > 3 and seg[3] in stop:
        del seg[:4]
    #print(seg)
    cre_exp_n.append(seg)

del cre_exp_n[1442][:3]
del cre_exp_n[1694][:3]
#for c in cre_exp_n:print(c)

sc = list(zip(q_id, q_inc, q_ques, q_fom, glu_exp_n, a1c_exp, cre_exp_n))

#for s in sc:print(s)
# ('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q1.csv')
df = pd.DataFrame(sc, columns=['q_id', 'Inclusion','Question','Formalized Expression', 'Glucose Statement',
                               'HbA1c Statement','Creatinine Statement'])

df.to_csv('/Users/luyu/Documents/Master Thesis/diabetes_criteria_test_postprocessing.csv', index=False)