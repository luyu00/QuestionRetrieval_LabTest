import re, nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import file as ufile
import pandas as pd

# let pandas dataframes show all rows
pd.set_option('display.max_columns', None)

# remove punctuation
def trans(s):
  exclude = string.punctuation
  return s.translate(str.maketrans({key: None for key in exclude}))

# for non-binaries or numbers not between 0 and 1, do min-max normalization then times weight
def min_max_normalization(trr):
  base = min(trr)
  range = max(trr) - base
  if range == 0:
      normalized = [0 for x in trr]
      return normalized
  else:
      normalized = [(x - base) / range for x in trr]
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
    words_stop = [str(c) for c in stopwords.words('english')]
    cat_num_stopw = []
    for c in cat:
        cat_num_stopw.append(len([w for w in words_stop if w in c.split()]))
    return cat_num_stopw

# count 'WH' question type
def count_WH_question(cat):
  cat_WH_question = []
  cat_WH = 'what|whats|how|when|why'.split('|')
  for c in cat:
    cat_WH_question.append(len([f.lower() for f in c.split() if f.lower() in cat_WH]))
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

def get_glucose_rang(exp):
    lab_range = []
    for c in exp:
        ch = ''.join([cc for cc in c])
        ch = ch.replace('[', '')
        ch = ch.replace(']', '')
        ch = ch.replace('"', '')
        ch = ch.replace('\'', '')
        ch = ch.replace(',', '')
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
        elif len(seg) > 2:
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
            elif float(seg[2]) > 125.0:
                lab_range.append(3)
        else:
            lab_range.append(0)
    return lab_range

def get_a1c_range(exp):
    lab_range = []
    for c in exp:
        ch = ''.join([cc for cc in c])
        ch = ch.replace('[', '')
        ch = ch.replace(']', '')
        ch = ch.replace('"', '')
        ch = ch.replace('\'', '')
        ch = ch.replace(',', '')
        ch = re.sub('(?<=\w)(\,)', '', ch)
        ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
        seg = ch.split()
        if len(seg) == 0:
            lab_range.append(0)
        elif len(seg) > 2:
            if float(seg[2]) < 5.7:
                lab_range.append(1)
            elif float(seg[2]) >= 5.7 and float(seg[2]) < 6.4:
                lab_range.append(2)
            elif float(seg[2]) >= 6.4:
                lab_range.append(3)
    return lab_range

def get_creatinine_range(exp):
    lab_range = []
    for c in exp:
        ch = ''.join([cc for cc in c])
        ch = ch.replace('[', '')
        ch = ch.replace(']', '')
        ch = ch.replace('"', '')
        ch = ch.replace('\'', '')
        ch = ch.replace(',', '')
        ch = ch.replace('Creatinine ratio', 'Creatinine_ratio')
        ch = ch.replace('serum creatinine level', 'serum_creatinine_level')
        ch = re.sub('(?<=\w)(\,)', '', ch)
        ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
        seg = ch.split()
        if len(seg) == 0:
            lab_range.append(0)
        elif len(seg) > 2:
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
    return lab_range

# load question pool
# load data: Glucose
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q10.csv')

# load data: HbA1c
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q10.csv')

# load data: Creatinine
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q10.csv')

# load data: no lab
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/noLab/noLab_Q10.csv')

# load data: lab mixed
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q10.csv')

#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q10.csv')

#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q10.csv')

data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q1.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q2.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q3.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q4.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q5.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q6.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q7.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q8.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q9.csv')
#data = ufile.read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q10.csv')

# load post-processed exp
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q1.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q2.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q3.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q4.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q5.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q6.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q7.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q8.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q9.csv")
#data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/Glucose_top10mix_post_Q10.csv")


# extract question
id = [d[0] for d in data]
candidates = [d[1] for d in data]
# extract lab-value statement for 3 lab tests
glu_exp = [d[2] for d in data]
a1c_exp = [d[3] for d in data]
cre_exp = [d[4] for d in data]
#for c in candidate:print(c)
#for g in glu_exp:print(g)
#for t in tfidf:print(t)

# extract features
lns = get_len(candidates)
n_stopw = count_stopword(candidates)
wh = count_WH_question(candidates)
glucose_test = get_lab_test(glu_exp)
glucose_range = get_glucose_rang(glu_exp)
hba1c_test = get_lab_test(a1c_exp)
hba1c_range = get_a1c_range(a1c_exp)
creatinine_test = get_lab_test(cre_exp)
creatinine_range = get_creatinine_range(cre_exp)

out = list(zip(lns, n_stopw, wh, glucose_test, glucose_range, hba1c_test, hba1c_range, creatinine_test, creatinine_range))
d_c = pd.DataFrame(out, columns=['Sentence length', 'Stopword count','WH question type',
                                 'Glucose test', 'Glucose range', 'HbA1c test', 'HbA1c range',
                                 'Creatinine test', 'Creatinine range'])
#d_c.to_csv('/Users/luyu/Documents/Master Thesis/test.csv', index=False)

# normalized features: for features that are non-binaries or not within range 0-1
lns_nm = min_max_normalization(get_len(candidates))
n_stopw_nm = min_max_normalization(count_stopword(candidates))
# wh
# glucose_test
glucose_range_nm = min_max_normalization(get_glucose_rang(glu_exp))
# hba1c_test
hba1c_range_nm = min_max_normalization(get_a1c_range(a1c_exp))
# creatinine_test
creatinine_range_nm = min_max_normalization(get_creatinine_range(cre_exp))

out = list(zip(lns_nm, n_stopw_nm, wh, glucose_test, glucose_range_nm, hba1c_test, hba1c_range_nm, creatinine_test, creatinine_range_nm))
d_d = pd.DataFrame(out, columns=['Sentence length', 'Stopword count','WH question type',
                                 'Glucose test', 'Glucose range', 'HbA1c test', 'HbA1c range',
                                 'Creatinine test', 'Creatinine range'])
#d_d.to_csv('/Users/luyu/Documents/Master Thesis/test.csv', index=False)

# compare feature difference between query q and candidate q
lth = [(1 - abs(d_d['Sentence length'][0] - x)) for x in d_d['Sentence length']]
sword = [(1 - abs(d_d['Stopword count'][0] - x)) for x in d_d['Stopword count']]
wh_q = wh
#glucose_test
#hba1c_test
#creatinine_test
glu_range = [0 if x == 0 else (1 - abs(d_d['Glucose range'][0] - x)) for x in d_d['Glucose range']]
a1c_range = [0 if x == 0 else (1 - abs(d_d['HbA1c range'][0] - x)) for x in d_d['HbA1c range']]
cre_range = [0 if x == 0 else (1 - abs(d_d['Creatinine range'][0] - x)) for x in d_d['Creatinine range']]
out = list(zip(id, candidates, glu_exp, a1c_exp, cre_exp, lth, sword, wh, glucose_test, glu_range, hba1c_test, a1c_range,
               creatinine_test, cre_range))
df = pd.DataFrame(out, columns=['ID', 'Question', 'Glucose Exp.', 'HbA1c Exp.', 'Creatinine Exp.',
                                'Sentence length', 'Stopword count','WH question type', 'Glucose test', 'Glucose range',
                                'HbA1c test', 'HbA1c range', 'Creatinine test', 'Creatinine range'])
#df.to_csv('/Users/luyu/Documents/Master Thesis/test.csv', index=False)
print(df)
