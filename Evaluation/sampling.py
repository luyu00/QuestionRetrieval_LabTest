import file as ufile
from random import sample
import pandas as pd
import re

# sample questions with 5-20 word length
data = ufile.read_csv('/Users/luyu/Downloads/ValX_demo/diabetes_criteria_test.csv')
d_20w = []

for d in data:
    d[1] = d[1].replace('\n', '')
    if 5 <= len(d[1].split()) <= 20:
        d_20w.append(d)

# write data
#ufile.write_csv('/Users/luyu/Documents/Master Thesis/diabetes_criteria_test_5to20words.csv', d_20w)

# count valx-parsed questions with expressions
box = ufile.read_csv('/Users/luyu/Documents/Master Thesis/diabetes_criteria_test_5to20words_parsed.csv')

q_id = [d[0] for d in box]
txt = [d[2] for d in box]
glu = [d[4] for d in box]
a1c = [d[5] for d in box]
cre = [d[6] for d in box]

with_exp = []
no_exp = []

for b in box:
    b[4] = b[4].replace('[', '')
    b[4] = b[4].replace(']', '')
    b[4] = b[4].replace('"', '')
    b[4] = b[4].replace('\'', '')
    b[4] = b[4].replace('=,', '=')
    b[4] = b[4].replace('absolute Glucose', 'absolute_Glucose')
    b[4] = b[4].replace('average Glucose', 'average_Glucose')
    b[4] = b[4].replace('mean Glucose', 'mean_Glucose')
    b[4] = b[4].replace('average glucose', 'average_glucose')
    b[4] = b[4].replace('Glucose tolerance', 'Glucose_tolerance')
    b[4] = b[4].replace('serum glucose level', 'serum_glucose_level')
    b[4] = b[4].replace('plasma glucose result', 'plasma_glucose_result')
    b[4] = b[4].replace('blood glucose measurement', 'blood_glucose_measurement')
    b[4] = b[4].replace('elevates my glucose-level', 'elevates_my_glucose_level')
    b[4] = b[4].split()
    b[5] = b[5].replace('[', '')
    b[5] = b[5].replace(']', '')
    b[5] = b[5].replace('"', '')
    b[5] = b[5].replace('\'', '')
    b[5] = b[5].replace('=,', '=')
    b[5] = b[5].split()
    b[6] = b[6].replace('[', '')
    b[6] = b[6].replace(']', '')
    b[6] = b[6].replace('"', '')
    b[6] = b[6].replace('\'', '')
    b[6] = b[6].replace('=,', '=')
    b[6] = b[6].replace('Creatinine ratio', 'Creatinine_ratio')
    b[6] = b[6].replace('serum creatinine level', 'serum_creatinine_level')
    b[6] = re.sub('(?<=\w)(\,)', '', b[6])
    b[6] = re.sub('(?<=\s)(\,)(?=\s)', '', b[6])
    b[6] = b[6].split()
    if len(b[4]) > 0 or len(b[5]) > 0 or len(b[6]) > 0:
        with_exp.append([b[0], b[2], b[4], b[5], b[6]])
    if len(b[4]) == 0 and len(b[5]) == 0 and len(b[6]) == 0:
        no_exp.append([b[0], b[2], b[4], b[5], b[6]])

#ufile.write_csv('/Users/luyu/Documents/Master Thesis/5to20w_with_exp.csv', with_exp)
#ufile.write_csv('/Users/luyu/Documents/Master Thesis/5to20w_no_exp.csv', no_exp)

# question pool has 3 components: valx_exo, no exp, other
# for above 2 csv files, replace with original text
ori = ufile.read_csv('/Users/luyu/Documents/Master Thesis/diabetes_criteria_test_5to20words.csv')

ori_id = [d[0] for d in ori]
ori_txt = [d[1] for d in ori]

for w in with_exp:
    w[1] = str([c[1] for c in ori if w[0] == c[0]])
    w[1] = w[1].replace('[', '')
    w[1] = w[1].replace(']', '')
    w[1] = w[1].replace('\'', '')
    w[1] = w[1].replace('"', '')

for w in no_exp:
    w[1] = str([c[1] for c in ori if w[0] == c[0]])
    w[1] = w[1].replace('[', '')
    w[1] = w[1].replace(']', '')
    w[1] = w[1].replace('\'', '')
    w[1] = w[1].replace('"', '')

# remove duplicates
with_exp = [with_exp[w] for w in range(len(with_exp)) if w == 0 or with_exp[w] != with_exp[w-1]]
no_exp = [no_exp[w] for w in range(len(no_exp)) if w == 0 or no_exp[w] != no_exp[w-1]]

#ufile.write_csv('/Users/luyu/Documents/Master Thesis/5to20w_with_exp.csv', with_exp)
#ufile.write_csv('/Users/luyu/Documents/Master Thesis/5to20w_no_exp.csv', no_exp)

# get other (questions not in the parsed file)
other = []
for o in ori:
    if o[0] not in q_id:
        other.append([o[0], o[1]])

#ufile.write_csv('/Users/luyu/Documents/Master Thesis/5to20w_other.csv', other)
# randomly sampling: all from with_exp, 1,347 from no_exp, and 1,348 from other
no_exp_sp = sample(no_exp, 1347)
other_sp = sample(other, 1348)

pool = with_exp + no_exp_sp + other_sp
ufile.write_csv('/Users/luyu/Documents/Master Thesis/Question_Pool.csv', pool)

# check question pool
q_pool = pd.read_csv('/Users/luyu/Documents/Master Thesis/Question_Pool.csv')
#print(q_pool)