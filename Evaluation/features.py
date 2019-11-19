import re, nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import file as ufile
import pandas as pd

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
    return min_max_normalization(cat_num_stopw)

# count 'WH' question type
def count_WH_question(cat):
  cat_WH_question = []
  cat_WH = 'what|whats|how|when|why'.split('|')
  for c in cat:
    cat_WH_question.append(len([f for f in c.split() if f.lower() in cat_WH]))
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
    return min_max_normalization(lab_range)

def get_a1c_range(exp):
    lab_range = []
    for c in exp:
        ch = ''.join([cc for cc in c])
        ch = ch.replace('[', '')
        ch = ch.replace(']', '')
        ch = ch.replace('"', '')
        ch = ch.replace('\'', '')
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
    return min_max_normalization(lab_range)

def get_creatinine_range(exp):
    lab_range = []
    for c in exp:
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
    return min_max_normalization(lab_range)


# load data: Creatinine
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q1.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q2.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q3.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q4.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q5.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q6.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q7.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q8.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q9.csv')
candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/creatinine/creatinine_Q10.csv')

# load data: HbA1c
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q1.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q2.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q3.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q4.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q5.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q6.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q7.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q8.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q9.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/hba1c/hba1c_Q10.csv')

# load data: Glucose
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q1.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q2.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q3.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q4.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q5.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q6.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q7.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q8.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q9.csv')
#candidates_d = ufile.read_csv('/Users/luyu/Documents/Master Thesis/glucose/glucose_Q10.csv')

# extract question id
candidates_id = [d[0] for d in candidates_d]

# extract question
candidates = [d[2] for d in candidates_d]

# extract expression for 3 lab
candidates_glu_exp = [d[4] for d in candidates_d]
candidates_a1c_exp = [d[5] for d in candidates_d]
candidates_cre_exp = [d[6] for d in candidates_d]

# extract normalized features
lns = min_max_normalization(get_len(candidates))
n_stopw = count_stopword(candidates)
wh = count_WH_question(candidates)
n_noun = num_of_noun(candidates)
n_vb = num_of_verb(candidates)
n_adj = num_of_adj(candidates)
glucose_test = get_lab_test(candidates_glu_exp)
glucose_range = get_glucose_rang(candidates_glu_exp)
hba1c_test = get_lab_test(candidates_a1c_exp)
hba1c_range = get_a1c_range(candidates_a1c_exp)
creatinine_test = get_lab_test(candidates_cre_exp)
creatinine_range = get_creatinine_range(candidates_cre_exp)

out = list(zip(lns, n_stopw, wh, n_noun, n_vb, n_adj, glucose_test, glucose_range, hba1c_test, hba1c_range, creatinine_test, creatinine_range))
d_d = pd.DataFrame(out, columns=['Sentence length', 'Stopword count','WH question type',
                                 'Number of nouns', 'Number of verbs','Number of adjectives',
                                 'Glucose test', 'Glucose range', 'HbA1c test', 'HbA1c range',
                                 'Creatinine test', 'Creatinine range'])

#d_d.to_csv('/Users/luyu/Documents/Master Thesis/test.csv', index=False)

# compare feature difference between query q and candidate q
lth = [(1 - abs(d_d['Sentence length'][0] - x)) for x in d_d['Sentence length']]
sword = [(1 - abs(d_d['Stopword count'][0] - x)) for x in d_d['Stopword count']]
wh_q = wh
n_noun = [(1 - abs(d_d['Number of nouns'][0] - x)) for x in d_d['Number of nouns']]
n_vb = [(1 - abs(d_d['Number of verbs'][0] - x)) for x in d_d['Number of verbs']]
n_adj = [(1 - abs(d_d['Number of adjectives'][0] - x)) for x in d_d['Number of adjectives']]
#glucose_test
#hba1c_test
#creatinine_test
glu_range = [0 if x == 0 else (1 - abs(d_d['Glucose range'][0] - x)) for x in d_d['Glucose range']]
a1c_range = [0 if x == 0 else (1 - abs(d_d['HbA1c range'][0] - x)) for x in d_d['HbA1c range']]
cre_range = [0 if x == 0 else (1 - abs(d_d['Creatinine range'][0] - x)) for x in d_d['Creatinine range']]
out = list(zip(lth, sword, wh, n_noun, n_vb, n_adj, glucose_test, glu_range, hba1c_test, a1c_range,
               creatinine_test, cre_range))
df = pd.DataFrame(out, columns=['Sentence length', 'Stopword count','WH question type',
                                 'Number of nouns', 'Number of verbs','Number of adjectives',
                                 'Glucose test', 'Glucose range', 'HbA1c test', 'HbA1c range',
                                 'Creatinine test', 'Creatinine range'])

#df.to_csv('/Users/luyu/Documents/Master Thesis/test.csv', index=False)