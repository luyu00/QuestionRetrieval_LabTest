import matplotlib.pyplot as plt
import os, re, csv, logging, nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import file as ufile
import numpy as np
import pandas as pd

# remove punctuation
def trans(s):
  exclude = string.punctuation
  return s.translate(str.maketrans({key: None for key in exclude}))

# for non-binaries or numbers not between 0 and 1, do min-max normalization then times weight
def min_max_normalization(trr):
  base = min(trr)
  range = max(trr) - base
  normalized = [(x-base)/range for x in trr]
  return normalized

# calculate features
def get_len(lab):
  lth = []
  for f in lab:
    lth.append(len(f.split()))
  return lth

# count number of question marks
def num_of_q_mark(cat):
  l = []
  for f in cat:
    l.append(len([ff for ff in f.split() if '?' in ff]))
  return l

# count number of stopwords
def count_stopword(cat):
  cat_num_stopw = []
  for c in cat:
    cat_num_stopw.append(len([w for w in words_stop if w in c.split()]))
  return min_max_normalization(cat_num_stopw)

# count 'WH' question type
def count_WH_question(cat):
  cat_WH_question = []
  cat_WH = 'who|what|how|where|when|why|which|whom|whose'.split('|')
  for c in cat:
    cat_WH_question.append(len([f for f in c.split() if f in cat_WH]))
    wh_cat = [i if i == 0 else 1 for i in cat_WH_question]
  return wh_cat

# get number of nouns
def num_of_noun(cat):
  cat_pos_tag = []
  for f in cat:
    cat_pos_tag.append(len([ff[1] for ff in nltk.pos_tag(f.split()) if 'NN' in ff[1]]))
  return min_max_normalization(cat_pos_tag)

# get number of verbs
def num_of_verb(cat):
  cat_pos_tag = []
  for f in cat:
    cat_pos_tag.append(len([ff[1] for ff in nltk.pos_tag(f.split()) if 'VB' in ff[1]]))
  return min_max_normalization(cat_pos_tag)

# get number of adjectives
def num_of_adj(cat):
  cat_pos_tag = []
  for f in cat:
    cat_pos_tag.append(len([ff[1] for ff in nltk.pos_tag(f.split()) if 'JJ' in ff[1]]))
  return min_max_normalization(cat_pos_tag)


# load data
creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q1.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q2.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q3.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q4.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q5.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q6.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q7.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q8.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q9.csv')
#creatinine_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q10.csv')



#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q1.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q2.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q3.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q4.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q5.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q6.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q7.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q8.csv')
#hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q9.csv')
hba1c_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q10.csv')

#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q1.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q2.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q3.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q4.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q5.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q6.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q7.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q8.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q9.csv')
glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q10.csv')

# ---------------------------------------
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q1.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q2.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q3.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q4.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q5.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q6.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q7.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q8.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q9.csv')
#glucose_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_mod/glucose_mod_Q10.csv')
# ---------------------------------------

creatinine_id = [d[0] for d in creatinine_d]
hba1c_id = [d[0] for d in hba1c_d]
glucose_id = [d[0] for d in glucose_d]

creatinine = [d[2] for d in creatinine_d]
hba1c = [d[2] for d in hba1c_d]
glucose = [d[2] for d in glucose_d]

creatinine_exp = [d[4] for d in creatinine_d]
hba1c_exp = [d[4] for d in hba1c_d]
glucose_exp = [d[4] for d in glucose_d]

# remove punctuation
creatinine_np = [trans(c) for c in creatinine]
hba1c_np = [trans(c) for c in hba1c]
glucose_np = [trans(c) for c in glucose]

# tokenization
creatinine_token = [word_tokenize(c) for c in creatinine_np]
hba1c_token = [word_tokenize(c) for c in hba1c_np]
glucose_token = [word_tokenize(c) for c in glucose_np]

# remove stopwords
words_stop = [str(c) for c in stopwords.words('english')]
creatinine_no_stopw = [[cc for cc in c if cc not in words_stop]for c in creatinine_token]
hba1c_no_stopw = [[cc for cc in c if cc not in words_stop]for c in hba1c_token]
glucose_no_stopw = [[cc for cc in c if cc not in words_stop]for c in glucose_token]

# stemming words
ps = nltk.stem.PorterStemmer()
creatinine_stemw = [[str(ps.stem(s)) for s in sw] for sw in creatinine_no_stopw]
hba1c_stemw = [[str(ps.stem(s)) for s in sw] for sw in hba1c_no_stopw]
glucose_stemw = [[str(ps.stem(s)) for s in sw] for sw in glucose_no_stopw]

# convert tokens back to sentence
creatinine_stemw_s = [' '.join(i) for i in creatinine_stemw]
hba1c_stemw_s = [' '.join(i) for i in hba1c_stemw]
glucose_stemw_s = [' '.join(i) for i in glucose_stemw]

# Creatinine: calculate normalized features

lns = min_max_normalization(get_len(creatinine))
num_q = [n for n in num_of_q_mark(creatinine)]
n_stopw = count_stopword(creatinine)
wh = count_WH_question(creatinine)
n_noun = num_of_noun(creatinine)
n_vb = num_of_verb(creatinine)
n_adj = num_of_adj(creatinine)

"""
# HbA1c: calculate normalized features

lns = min_max_normalization(get_len(hba1c))
num_q = [n for n in num_of_q_mark(hba1c)]
n_stopw = count_stopword(hba1c)
wh = min_max_normalization(count_WH_question(hba1c))
n_noun = num_of_noun(hba1c)
n_vb = num_of_verb(hba1c)
n_adj = num_of_adj(hba1c)
"""

# Glucose: calculate normalized features
"""
lns = min_max_normalization(get_len(glucose))
num_q = [n for n in num_of_q_mark(glucose)]
n_stopw = count_stopword(glucose)
wh = min_max_normalization(count_WH_question(glucose))
n_noun = num_of_noun(glucose)
n_vb = num_of_verb(glucose)
n_adj = num_of_adj(glucose)
"""

# calculate lab test feature for each lab
def get_lab_test(test):
    labo = []
    for c in test:
        ch = ''.join([cc for cc in c])
        ch = ch.replace('[', '')
        ch = ch.replace(']', '')
        ch = ch.replace('"', '')
        ch = ch.replace('\'', '')
        ch = re.sub('(?<=\w)(\,)', '', ch)
        ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
        seg = ch.split()
        if len(seg) != 0:
            labo.append(1)
        else:
            labo.append(0)
    return labo

lab = get_lab_test(creatinine_exp)
#lab = get_lab_test(hba1c_exp)
#lab = get_lab_test(glucose_exp)


# calculate lab range feature for Creatinine
lab_range = []
for c in creatinine_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
    ch = ch.replace('Creatinine ratio', 'Creatinine_ratio')
    ch = ch.replace('serum creatinine level', 'serum_creatinine_level')
    ch = re.sub('(?<=\w)(\,)', '', ch)
    ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
    seg = ch.split()
    if len(seg) == 0:
        lab_range.append(0)
    elif len(seg) != 0:
        if float(seg[2]) < 0.84:
            lab_range.append(1)
        elif float(seg[2]) >= 0.84 and float(seg[2]) <= 1.21:
            lab_range.append(2)
        elif float(seg[2]) > 1.21 and float(seg[2]) < 20.0:
            lab_range.append(3)
        elif float(seg[2]) > 20.0 and float(seg[2]) < 74.3:
            lab_range.append(1)
        elif float(seg[2]) >= 74.3 and float(seg[2]) <= 107.0:
            lab_range.append(2)
        elif float(seg[2]) > 107.0:
            lab_range.append(3)


"""
# calculate lab range feature for HbA1c
lab_range = []
for c in hba1c_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
    ch = ch.replace('Creatinine ratio', 'Creatinine_ratio')
    ch = ch.replace('serum creatinine level', 'serum_creatinine_level')
    ch = re.sub('(?<=\w)(\,)', '', ch)
    ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
    seg = ch.split()
    if len(seg) == 0:
        lab_range.append(0)
    elif len(seg) != 0:
        if float(seg[2]) < 5.7:
            lab_range.append(1)
        elif float(seg[2]) >= 5.7 and float(seg[2]) < 6.4:
            lab_range.append(2)
        else:
            lab_range.append(3)
"""
"""
# calculate lab range feature for Glucose
lab_range = []
for c in glucose_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
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
    if len(seg) == 0:
        lab_range.append(0)
    elif len(seg) != 0:
        if float(seg[2]) < 5.6:
            lab_range.append(1)
        elif 5.6 <= float(seg[2]) <= 6.9:
            lab_range.append(2)
        elif 6.9 < float(seg[2]) < 20.0:
            lab_range.append(3)
        elif 20.0 <= float(seg[2]) < 100.0:
            lab_range.append(1)
        elif 100.0 <= float(seg[2]) <= 125.0:
            lab_range.append(2)
        else:
            lab_range.append(3)
"""
lb_range = min_max_normalization(lab_range)

# pack all features into dataframe
out = list(zip(lns, num_q, n_stopw, wh, n_noun, n_vb, n_adj, lab, lb_range))
d_d = pd.DataFrame(out, columns=['Sentence length', 'number of \'?\'', 'Stopword count',
                               'WH question type', 'Number of nouns', 'Number of verbs',
                               'Number of adjectives','Lab test', 'Lab range'])

# compare diff between query q and candidate q
lth = [(1 - abs(d_d['Sentence length'][0] - x)) for x in d_d['Sentence length']]
qmark = [(1 - abs(d_d['number of \'?\''][0] - x)) for x in d_d['number of \'?\'']]
sword = [(1 - abs(d_d['Stopword count'][0] - x)) for x in d_d['Stopword count']]
wh_q = wh
n_noun = [(1 - abs(d_d['Number of nouns'][0] - x)) for x in d_d['Number of nouns']]
n_vb = [(1 - abs(d_d['Number of verbs'][0] - x)) for x in d_d['Number of verbs']]
n_adj = [(1 - abs(d_d['Number of adjectives'][0] - x)) for x in d_d['Number of adjectives']]
lab_test = lab
test_range = [(1 - abs(d_d['Lab range'][0] - x)) for x in d_d['Lab range']]
out = list(zip(lth, qmark, sword, wh, n_noun, n_vb, n_adj, lab_test, test_range))

# pack all feature diff into dataframe
df = pd.DataFrame(out, columns=['Sentence length', 'number of \'?\'', 'Stopword count',
                               'WH question type', 'Number of nouns', 'Number of verbs',
                               'Number of adjectives','Lab test', 'Lab range'])

#df.to_csv('/Users/luyu/Documents/Master Thesis/test.csv', index=False)

data = ufile.read_csv("/Users/luyu/Downloads/ValX_demo/diabetes_criteria_test_allWith3lab_exp.csv")

# get questions no longer then 20 words
"""
len20 = []

for d in data:
    if len(d[2].split()) <= 20:
        len20.append(d)

include = []
for l in len20:
    # post-processing glucose expression
    l[4] = ''.join([cc for cc in l[4]])
    l[4] = l[4].replace('[', '')
    l[4] = l[4].replace(']', '')
    l[4] = l[4].replace('"', '')
    l[4] = l[4].replace('\'', '')
    l[4] = re.sub('(?<=\w)(\,)', '', l[4])
    l[4] = re.sub('(?<=\s)(\,)(?=\s)', '', l[4])
    l[4] = l[4].split()
    l[5] = ''.join([cc for cc in l[5]])
    l[5] = l[5].replace('[', '')
    l[5] = l[5].replace(']', '')
    l[5] = l[5].replace('"', '')
    l[5] = l[5].replace('\'', '')
    l[5] = re.sub('(?<=\w)(\,)', '', l[5])
    l[5] = re.sub('(?<=\s)(\,)(?=\s)', '', l[5])
    l[5] = l[5].split()
    l[6] = l[6].replace('[', '')
    l[6] = l[6].replace(']', '')
    l[6] = l[6].replace('"', '')
    l[6] = l[6].replace('\'', '')
    l[5] = re.sub('(?<=\w)(\,)', '', l[6])
    l[6] = re.sub('(?<=\s)(\,)(?=\s)', '', l[6])
    #print(l)
    #if len(l[4]) != 0 or len(l[5]) != 0 or len(l[6]) != 0:
    if len(l[6]) != 0:
        include.append(l)
"""
# see questions include either 1 of the 3 lab
#print(len(include))
#print(len(len20))
#for l in len20:print(l)
"""
data = ufile.read_csv("/Users/luyu/Downloads/ValX_demo/diabetes_criteria_test.csv")
# calculate distribution of each WH question type
h = 0
i = 0
j = 0
k = 0
l = 0
m = 0
n = 0
o = 0
p = 0
for d in data:
    if 'who' in d[1].split():h+=1
    if 'what' in d[1].split(): i += 1
    if 'how' in d[1].split(): j += 1
    if 'where' in d[1].split(): k += 1
    if 'when' in d[1].split(): l += 1
    if 'why' in d[1].split(): m += 1
    if 'which' in d[1].split(): n += 1
    if 'whom' in d[1].split(): o += 1
    if 'whose' in d[1].split(): p += 1

print([h,i, j,k,l,m,n,o,p])
"""
#ufile.write_csv("/Users/luyu/Downloads/ValX_demo/diabetes_criteria_test_all20With3lab.csv", len20)

data = ufile.read_csv("/Users/luyu/Downloads/ValX_demo/diabetes_criteria_test_20words_3lab20words.csv")

include = []
for l in data:
    l[4] = l[4].replace('[', '')
    l[4] = l[4].replace(']', '')
    l[4] = l[4].replace('"', '')
    l[4] = l[4].replace('\'', '')
    l[4] = re.sub('(?<=\w)(\,)', '', l[4])
    l[4] = re.sub('(?<=\s)(\,)(?=\s)', '', l[4])
    l[5] = l[5].replace('[', '')
    l[5] = l[5].replace(']', '')
    l[5] = l[5].replace('"', '')
    l[5] = l[5].replace('\'', '')
    l[5] = re.sub('(?<=\w)(\,)', '', l[5])
    l[5] = re.sub('(?<=\s)(\,)(?=\s)', '', l[5])
    l[6] = l[6].replace('[', '')
    l[6] = l[6].replace(']', '')
    l[6] = l[6].replace('"', '')
    l[6] = l[6].replace('\'', '')
    l[6] = re.sub('(?<=\w)(\,)', '', l[6])
    l[6] = re.sub('(?<=\s)(\,)(?=\s)', '', l[6])
    #if len(l[4]) != 0 or len(l[5]) != 0 or len(l[6]) != 0:
    if len(l[6]) != 0:
        include.append(l)

#for l in include:print(l)