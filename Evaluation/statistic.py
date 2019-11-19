import re, nltk, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import file as ufile
import pandas as pd

stop = "gm|%|%,|hr|hrs|min|mins|minute|minutes|hour|hours|okay hour|day|days|week|weeks|month|months|yr|yrs|year|years".split("|")

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
    words_stop = [str(c) for c in stopwords.words('english')]
    cat_num_stopw = []
    for c in cat:
        cat_num_stopw.append(len([w for w in words_stop if w.lower() in c.split()]))
    return cat_num_stopw

# count 'WH' question type
def count_WH_question(cat):
  cat_WH_question = []
  cat_WH = 'what|how|when|why'.split('|')
  for c in cat:
    cat_WH_question.append(len([f for f in c.split() if f.lower() in cat_WH]))
    wh_cat = [i if i == 0 else 1 for i in cat_WH_question]
  return wh_cat

# count 'WH' question type
def count_what_question(cat):
  cat_WH_question = []
  cat_WH = 'what'
  for c in cat:
    cat_WH_question.append(len([f for f in c.split() if cat_WH in f]))
    wh_cat = [i if i == 0 else 1 for i in cat_WH_question]
  return wh_cat

def count_how_questions(cat):
    cat_WH_question = []
    cat_WH = 'how'
    for c in cat:
        cat_WH_question.append(len([f for f in c.split() if cat_WH in f]))
        wh_cat = [i if i == 0 else 1 for i in cat_WH_question]
    return wh_cat

def count_when_questions(cat):
    cat_WH_question = []
    cat_WH = 'when'
    for c in cat:
        cat_WH_question.append(len([f for f in c.split() if cat_WH in f]))
        wh_cat = [i if i == 0 else 1 for i in cat_WH_question]
    return wh_cat

def count_why_questions(cat):
    cat_WH_question = []
    cat_WH = 'why'
    for c in cat:
        cat_WH_question.append(len([f for f in c.split() if cat_WH in f]))
        wh_cat = [i if i == 0 else 1 for i in cat_WH_question]
    return wh_cat

"""
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
"""

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

def get_glucose_rang(segm):
    lab_range = []
    for seg in segm:
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
            if 0 < float(seg[2]) < 5.7:
                lab_range.append(1)
            elif 5.7 <= float(seg[2]) < 6.4:
                lab_range.append(2)
            elif 6.4 <= float(seg[2]) <= 11:
                lab_range.append(3)
            else:
                lab_range.append(0)
    return lab_range

def get_creatinine_range(segm):
    lab_range = []
    for seg in segm:
        if len(seg) == 0:
            lab_range.append(0)
        elif len(seg) > 2:
            if float(seg[2]) < 0.84:
                lab_range.append(1)
            elif 0.84 <= float(seg[2]) <= 1.21:
                lab_range.append(2)
            elif 1.21 < float(seg[2]) < 20.0:
                lab_range.append(3)
            elif 20.0 < float(seg[2]) < 74.3:
                lab_range.append(1)
            elif 74.3 <= float(seg[2]) <= 107.0:
                lab_range.append(2)
            elif float(seg[2]) > 107.0:
                lab_range.append(3)
            else:
                lab_range.append(0)
    return lab_range

# load data
pool = ufile.read_csv('/Users/luyu/Documents/Master Thesis/Question_Pool.csv')
#del pool[0]

# extract expression for 3 lab
id = [d[0] for d in pool]
txt = [d[1] for d in pool]
glu_exp = [d[2] for d in pool]
a1c_exp = [d[3] for d in pool]
cre_exp = [d[4] for d in pool]

# post-processing glucose results
glu_exp_n = []
for c in glu_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
    ch = ch.replace('=,', '=')
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
    glu_exp_n.append(seg)

#for g in glu_exp_n:print([type(gg) for gg in g])

# post-processing creatinine results
cre_exp_n = []
for c in cre_exp:
    ch = ''.join([cc for cc in c])
    ch = ch.replace('[', '')
    ch = ch.replace(']', '')
    ch = ch.replace('"', '')
    ch = ch.replace('\'', '')
    ch = ch.replace('=,', '=')
    ch = ch.replace(',', '')
    ch = ch.replace('Creatinine ratio', 'Creatinine_ratio')
    ch = ch.replace('serum creatinine level', 'serum_creatinine_level')
    ch = re.sub('(?<=\w)(\,)', '', ch)
    ch = re.sub('(?<=\s)(\,)(?=\s)', '', ch)
    seg = ch.split()
    cre_exp_n.append(seg)

lns = get_len(txt)
n_stopw = count_stopword(txt)
wh = count_WH_question(txt)
#what = count_what_question(parsed)
#how = count_how_questions(parsed)
#when = count_when_questions(parsed)
#why = count_why_questions(parsed)
glucose_test = get_lab_test(glu_exp)
glucose_range = get_glucose_rang(glu_exp_n)
hba1c_test = get_lab_test(a1c_exp)
hba1c_range = get_a1c_range(a1c_exp)
creatinine_test = get_lab_test(cre_exp)
creatinine_range = get_creatinine_range(cre_exp_n)

"""
chang = []
for g in glucose_range:
    if g != 0:
        chang.append(g)
print(len(chang))
"""

out = list(zip(id, txt, glu_exp_n, a1c_exp, cre_exp, lns, n_stopw, wh, glucose_test, glucose_range, hba1c_test, hba1c_range, creatinine_test,
               creatinine_range))

df = pd.DataFrame(out, columns=['ID', 'Question', 'Glucose Exp', 'A1c Exp', 'Creatinine Exp', 'Sentence length', 'Stopword count','WH questions', 'Glucose test', 'Glucose range', 'HbA1c test',
                                'HbA1c range', 'Creatinine test', 'Creatinine range'])

df.to_csv('/Users/luyu/Documents/Master Thesis/feature statistic.csv', index=False)
"""
tests = zip(glucose_test, hba1c_test, creatinine_test)
no_lab = 0
for t in tests:
    if (0, 0, 0) == t:
        no_lab += 1
print(no_lab)
"""
"""
fea = ufile.read_csv('/Users/luyu/Documents/Master Thesis/fea_values.csv')
glu_range = [d[7] for d in fea]
a1c_range = [d[9] for d in fea]
cre_range = [d[11] for d in fea]

glu_range = glu_range[1:]
a1c_range = a1c_range[1:]
cre_range = cre_range[1:]

nn = 0
lo = 0
nm = 0
hi = 0

for d in cre_range:
    if int(d) ==1:
        lo += 1
    elif int(d) ==2:
        nm += 1
    if int(d) == 3:
        hi += 1
    else:
        nn +=1

print(lo, nm, hi, nn)
"""
