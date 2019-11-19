import file as ufile
import numpy as np

# calculate DCG
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


# calculate nDCG
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


# load ranked data
# Creatinine
data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q1.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q2.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q3.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q4.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q5.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q6.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q7.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q8.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q9.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q10.csv")[1:]
# HbA1c

# Glucose

# no lab

"""
# extract id and question
id = [d[0] for d in data if d[0] is not '']
candidates = [d[1] for d in data if d[1] is not '']
# extract lab-value statement for 3 lab tests
glu_exp = [d[2] for d in data if d[2] is not '']
a1c_exp = [d[3] for d in data if d[3] is not '']
cre_exp = [d[4] for d in data if d[4] is not '']
# extract vector-space similarities
tfidf = [float(d[5]) for d in data if d[5] is not '']
elmo = [float(d[6]) for d in data if d[6] is not '']
use = [float(d[7]) for d in data if d[7] is not '']
# extract feature distances
lth = [float(d[8]) for d in data if d[8] is not '']
sword = [float(d[9]) for d in data if d[9] is not '']
wh = [float(d[10]) for d in data if d[10] is not '']
# extract lab test and range
lab_test = [float(d[11]) for d in data if d[12] is not '']
lab_range = [float(d[12]) for d in data if d[12] is not '']
"""
# extract gold-standard rating
gs = [float(d[13]) for d in data if d[13] is not '']

# use gold-standard rating as rel score to calculate nDCG
nDCG = []
for i in range(1, 11):
    print(ndcg_at_k(gs, i))
    #nDCG.append(ndcg_at_k(gs, i))
