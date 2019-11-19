import file as ufile
import pandas as pd
import numpy as np

# let pandas dataframes show all rows
pd.set_option('display.max_columns', None)


# define a function that calculates the tfidf-weighted similarity score
def tfidf_lab_sim(tf, lab_t, lab_r):
    w_tf = 0.65
    w_lab_t = 0.16
    w_lab_r = 0.19
    tot = w_tf*tf + w_lab_t*lab_t + w_lab_r*lab_r
    return tot


# define a function that calculates the elmo-weighted similarity score
def elmo_lab_sim(eo, lab_t, lab_r):
    w_eo = 0.65
    w_lab_t = 0.16
    w_lab_r = 0.19
    tot = w_eo*eo + w_lab_t*lab_t + w_lab_r*lab_r
    return tot

# define a function that calculates the tfidf-weighted similarity score
def use_lab_sim(us, lab_t, lab_r):
    w_us = 0.65
    w_lab_t = 0.16
    w_lab_r = 0.19
    tot = w_us*us + w_lab_t*lab_t + w_lab_r*lab_r
    return tot


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


# load data
# load data: Creatinine
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q1.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q2.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q3.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q4.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q5.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q6.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q7.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q8.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q9.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Creatinine_top10mix_sp/Creatinine_top10mix_sp_Q10.csv")[1:]

# load data: HbA1c
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q1.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q2.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q3.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q4.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q5.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q6.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q7.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q8.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q9.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/HbA1c_top10mix_sp/HbA1c_top10mix_sp_Q10.csv")[1:]

# load data: Glucose
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q1.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q2.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q3.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q4.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q5.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q6.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q7.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q8.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q9.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Glucose_top10mix_sp_Q10.csv")[1:]

# load data: CE Glucose
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q1.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q2.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q3.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q4.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q5.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q6.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q7.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q8.csv")[1:]
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q9.csv")[1:]
data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/lab_10q_sp/Glucose_top10mix_sp/Correctly Extracted/Glucose_top10mix_cesp_Q10.csv")[1:]

# extract id and question
id = [d[0] for d in data if d[0] is not '']
candidates = [d[1] for d in data if d[1] is not '']
# extract lab-value statement for 3 lab tests
glu_exp = [d[2] if d[2] is not '' else '[]' for d in data]
del glu_exp[1]
a1c_exp = [d[3] if d[3] is not '' else '[]' for d in data]
del a1c_exp[1]
cre_exp = [d[4] for d in data if d[4] is not '']
cre_exp = [d[4] if d[4] is not '' else '[]' for d in data]
del cre_exp[1]
# extract vector-space similarities
tfidf = [float(d[5]) for d in data if d[5] is not '']
elmo = [float(d[6]) for d in data if d[6] is not '']
use = [float(d[7]) for d in data if d[7] is not '']
lab_test = [float(d[11]) for d in data if d[12] is not '']
lab_range = [float(d[12]) for d in data if d[12] is not '']
# extract gold-standard rating
gs = [0] + [float(d[13]) for d in data if d[13] is not '']

# zip
out = list(zip(id, candidates, glu_exp, a1c_exp, cre_exp, tfidf, elmo, use, lab_test, lab_range))
# store as dataframe
df = pd.DataFrame(out, columns=['ID', 'Question', 'Glucose Exp.', 'HbA1c Exp.', 'Creatinine Exp.', 'TF-IDF', 'ELMo', 'USE',
                                'lab test', 'lab range'])

# calculate total similarity score
# tfidf + lab
#total_score = tfidf_lab_sim(df['TF-IDF'], df['lab test'], df['lab range'])
# elmo + lab
#total_score = elmo_lab_sim(df['ELMo'], df['lab test'], df['lab range'])
# use + lab
total_score = use_lab_sim(df['USE'], df['lab test'], df['lab range'])

# add 'total similarity score' and 'gold-standard rating' to dataframe
df['Total Score'] = total_score
df.insert(df.shape[1], column='Gold-Standard Rating', value=gs)

# sort candidates by total/single-model similarity scores
#df_new = pd.concat([df.head(1), df[1:].sort_values(by='TF-IDF', ascending=False)])
#df_new = pd.concat([df.head(1), df[1:].sort_values(by='ELMo', ascending=False)])
df_new = pd.concat([df.head(1), df[1:].sort_values(by='USE', ascending=False)])
#df_new = pd.concat([df.head(1), df[1:].sort_values(by='Total Score', ascending=False)])

# use gold-standard rating as rel score to calculate nDCG
reli = df_new['Gold-Standard Rating'].iloc[1:]
nDCG = []

#print(df_new['TF-IDF'], df_new['Gold-Standard Rating'])

# calculate nDCG@[1, 10]
for i in range(1, 11):
    print(ndcg_at_k(reli, i))
    #nDCG.append(ndcg_at_k(reli, i))

#Creatinine nDCG
# TF-IDF + lab
"""Q1: [0.8800000000095999, 0.7890490856082704, 0.8637516118156833, 0.8282093100734655, 0.8939014528147818, 
0.9411331664978762, 0.9097437172887339, 0.9442471407564559, 0.9450937186214871, 0.9454911137632452]"""
