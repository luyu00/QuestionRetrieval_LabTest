import re, nltk, string
from nltk.corpus import stopwords
import file as ufile
import pandas as pd

# define a group of impossible units
stop = "gm|hr|hrs|min|mins|minute|minutes|hour|hours|okay hour|day|days|week|weeks|month|months|yr|yrs|year|years".split("|")

data = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/exp/Glucose_top10mix_exp_Q3.csv")
#ini = ufile.read_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_top10mix_csv/Glucose_top10mix_Q1.csv")

glu_exp = [d[2] for d in data]

out = []
# remove expressions containing impossible units
for d in data:
    #print(d[0], d[2], d[3], d[4])
    for s in stop:
        if s in d[2]:
            d[2] = '[' + d[2].split(s)[1][:-2] +']'
            d[2] = d[2].replace("s,',", "")
    out.append([d[0], d[1], d[2], d[3], d[4]])
    #print(d[0], d[2])

ufile.write_csv("/Users/luyu/Documents/Master Thesis/rating/mix_top10/Glucose_top10mix/Glucose_exp/post_exp/test.csv", out)