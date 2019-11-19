import file as ufile
import pandas as pd
import numpy as np

# let pandas dataframes show all rows
pd.set_option('display.max_columns', None)


# define a function that calculates the total similarity score
def total_sim_noLab_fea(tf, w_tf, eo, w_eo, us, w_us, lnn, w_lnn, swc, w_swc, whq, w_whq):
    """
    w_tf = 0.15
    w_eo = 0.2
    w_us = 0.3
    w_lnn = 0.08
    w_swc = 0.05
    w_whq = 0.05
    w_lab_t = 0.1
    w_lab_r = 0.07
    """
    tot = w_tf*tf + w_eo*eo + w_us*us + w_lnn*lnn + w_swc*swc + w_whq*whq
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
cre_exp = [d[4] if d[4] is not '' else '[]' for d in data]
del cre_exp[1]
# extract vector-space similarities
tfidf = [float(d[5]) for d in data if d[5] is not '']
elmo = [float(d[6]) for d in data if d[6] is not '']
use = [float(d[7]) for d in data if d[7] is not '']
# extract feature distances
lth = [float(d[8]) for d in data if d[8] is not '']
sword = [float(d[9]) for d in data if d[9] is not '']
wh = [float(d[10]) for d in data if d[10] is not '']
# extract lab test and range
#lab_test = [float(d[11]) for d in data if d[12] is not '']
#lab_range = [float(d[12]) for d in data if d[12] is not '']
# extract gold-standard rating
gs = [0] + [float(d[13]) for d in data if d[13] is not '']

# zip
out = list(zip(id, candidates, glu_exp, a1c_exp, cre_exp, tfidf, elmo, use, lth, sword, wh))

# store as dataframe
# creatinine
df = pd.DataFrame(out, columns=['ID', 'Question', 'Glucose Exp.', 'HbA1c Exp.', 'Creatinine Exp.', 'TF-IDF', 'ELMo', 'USE',
                                'Sentence length', 'Stopword count','WH question type'])
# calculate total similarity score
"""
    w_tf = 0.15
    w_eo = 0.2
    w_us = 0.3
    w_lnn = 0.08
    w_swc = 0.05
    w_whq = 0.05
    w_lab_t = 0.1
    w_lab_r = 0.07
    """
total_score = total_sim_noLab_fea(df['TF-IDF'], 0.15, df['ELMo'], 0.2, df['USE'], 0.3, df['Sentence length'], 0.08,
                                  df['Stopword count'], 0.05, df['WH question type'], 0.05)
# add 'total similarity score' and 'gold-standard rating' to dataframe
df['Total Score'] = total_score
df.insert(df.shape[1], column='Gold-Standard Rating', value=gs)

# sort candidates by total similarity scores
df_new = pd.concat([df.head(1), df[1:].sort_values(by='Total Score', ascending=False)])
# use gold-standard rating as rel score to calculate nDCG
reli = df_new['Gold-Standard Rating'].iloc[1:]
nDCG = []

# calculate nDCG@[1, 10]
for i in range(1, 11):
    print(ndcg_at_k(reli, i))
    #nDCG.append(reli, i))

#Creatinine nDCG
"""Q1: [1.0, 0.9685653521836393, 0.8189002654276691, 0.8614171123398141, 0.8737694963541588, 
0.8327750408526775, 0.8780104257436057, 0.9132829663670886, 0.9070074817246354, 0.9511067280608354]
"""
