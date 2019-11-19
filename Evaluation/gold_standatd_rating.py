import numpy as np
import pandas as pd
import file as ufile


# get gold stadard by averaging 3 raters' scores
def get_gold_standard(r1, r2, r3):
    gold_standard = [(x+y+z)/3 for x, y, z in zip(r1, r2, r3)]
    return gold_standard


# load file and extract similarity score
def load_score(dir):
    fo = ufile.read_csv(dir)
    del fo[:3]
    nm = [int(d[2]) for d in fo]
    return nm


def dcg_at_k(relevance, rank):
    relevance = np.asarray(relevance)[:rank]
    #print(relevance, '\n')
    n_relevance = len(relevance)
    if n_relevance == 0:
        return 0
    discount = np.log2(np.arange(n_relevance) + 2)
    return np.sum(relevance / discount)


def ndcg_at_k(relevance, rank):
    best_dcg = dcg_at_k(sorted(relevance, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg_at_k(relevance, rank) / best_dcg

#print(ndcg_at_k([3, 2, 5, 3, 0, 0, 1, 4, 2, 2, 3, 0], 3))

# extract similarity scores for lab_allModels
score_fu_cre_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q1.csv')
score_fu_cre_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q2.csv')
score_fu_cre_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q3.csv')
score_fu_cre_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q4.csv')
score_fu_cre_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q5.csv')
score_fu_cre_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q6.csv')
score_fu_cre_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q7.csv')
score_fu_cre_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q8.csv')
score_fu_cre_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q9.csv')
score_fu_cre_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_allModels_top10_Q10.csv')

score_fu_glu_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q1.csv')
score_fu_glu_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q2.csv')
score_fu_glu_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q3.csv')
score_fu_glu_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q4.csv')
score_fu_glu_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q5.csv')
score_fu_glu_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q6.csv')
score_fu_glu_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q7.csv')
score_fu_glu_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q8.csv')
score_fu_glu_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q9.csv')
score_fu_glu_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_allModels_top10_Q10.csv')

score_fu_a1c_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q1.csv')
score_fu_a1c_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q2.csv')
score_fu_a1c_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q3.csv')
score_fu_a1c_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q4.csv')
score_fu_a1c_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q5.csv')
score_fu_a1c_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q6.csv')
score_fu_a1c_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q7.csv')
score_fu_a1c_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q8.csv')
score_fu_a1c_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q9.csv')
score_fu_a1c_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_allModels_top10_Q10.csv')

score_fu_noLab_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q1.csv')
score_fu_noLab_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q2.csv')
score_fu_noLab_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q3.csv')
score_fu_noLab_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q4.csv')
score_fu_noLab_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q5.csv')
score_fu_noLab_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q6.csv')
score_fu_noLab_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q7.csv')
score_fu_noLab_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q8.csv')
score_fu_noLab_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q9.csv')
score_fu_noLab_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_allModels_top10_Q10.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

score_zou_cre_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q1.csv')
score_zou_cre_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q2.csv')
score_zou_cre_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q3.csv')
score_zou_cre_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q4.csv')
score_zou_cre_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q5.csv')
score_zou_cre_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q6.csv')
score_zou_cre_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q7.csv')
score_zou_cre_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q8.csv')
score_zou_cre_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q9.csv')
score_zou_cre_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_allModels_top10_Q10.csv')

score_zou_glu_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q1.csv')
score_zou_glu_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q2.csv')
score_zou_glu_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q3.csv')
score_zou_glu_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q4.csv')
score_zou_glu_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q5.csv')
score_zou_glu_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q6.csv')
score_zou_glu_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q7.csv')
score_zou_glu_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q8.csv')
score_zou_glu_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q9.csv')
score_zou_glu_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_allModels_top10_Q10.csv')

score_zou_a1c_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q1.csv')
score_zou_a1c_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q2.csv')
score_zou_a1c_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q3.csv')
score_zou_a1c_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q4.csv')
score_zou_a1c_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q5.csv')
score_zou_a1c_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q6.csv')
score_zou_a1c_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q7.csv')
score_zou_a1c_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q8.csv')
score_zou_a1c_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q9.csv')
score_zou_a1c_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_allModels_top10_Q10.csv')

score_zou_noLab_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q1.csv')
score_zou_noLab_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q2.csv')
score_zou_noLab_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q3.csv')
score_zou_noLab_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q4.csv')
score_zou_noLab_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q5.csv')
score_zou_noLab_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q6.csv')
score_zou_noLab_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q7.csv')
score_zou_noLab_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q8.csv')
score_zou_noLab_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q9.csv')
score_zou_noLab_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_allModels_top10_Q10.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

score_zhang_cre_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q1.csv')
score_zhang_cre_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q2.csv')
score_zhang_cre_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q3.csv')
score_zhang_cre_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q4.csv')
score_zhang_cre_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q5.csv')
score_zhang_cre_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q6.csv')
score_zhang_cre_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q7.csv')
score_zhang_cre_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q8.csv')
score_zhang_cre_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q9.csv')
score_zhang_cre_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_allModels_top10_Q10.csv')

score_zhang_glu_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q1.csv')
score_zhang_glu_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q2.csv')
score_zhang_glu_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q3.csv')
score_zhang_glu_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q4.csv')
score_zhang_glu_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q5.csv')
score_zhang_glu_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q6.csv')
score_zhang_glu_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q7.csv')
score_zhang_glu_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q8.csv')
score_zhang_glu_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q9.csv')
score_zhang_glu_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_allModels_top10_Q10.csv')

score_zhang_a1c_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q1.csv')
score_zhang_a1c_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q2.csv')
score_zhang_a1c_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q3.csv')
score_zhang_a1c_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q4.csv')
score_zhang_a1c_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q5.csv')
score_zhang_a1c_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q6.csv')
score_zhang_a1c_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q7.csv')
score_zhang_a1c_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q8.csv')
score_zhang_a1c_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q9.csv')
score_zhang_a1c_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_allModels_top10_Q10.csv')

score_zhang_noLab_allModels_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q1.csv')
score_zhang_noLab_allModels_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q2.csv')
score_zhang_noLab_allModels_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q3.csv')
score_zhang_noLab_allModels_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q4.csv')
score_zhang_noLab_allModels_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q5.csv')
score_zhang_noLab_allModels_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q6.csv')
score_zhang_noLab_allModels_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q7.csv')
score_zhang_noLab_allModels_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q8.csv')
score_zhang_noLab_allModels_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q9.csv')
score_zhang_noLab_allModels_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_allModels_top10_Q10.csv')

# extract similarity scores for lab_ELMo
score_fu_cre_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q1.csv')
score_fu_cre_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q2.csv')
score_fu_cre_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q3.csv')
score_fu_cre_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q4.csv')
score_fu_cre_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q5.csv')
score_fu_cre_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q6.csv')
score_fu_cre_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q7.csv')
score_fu_cre_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q8.csv')
score_fu_cre_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q9.csv')
score_fu_cre_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_elmo_top10_Q10.csv')

score_fu_glu_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q1.csv')
score_fu_glu_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q2.csv')
score_fu_glu_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q3.csv')
score_fu_glu_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q4.csv')
score_fu_glu_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q5.csv')
score_fu_glu_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q6.csv')
score_fu_glu_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q7.csv')
score_fu_glu_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q8.csv')
score_fu_glu_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q9.csv')
score_fu_glu_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_elmo_top10_Q10.csv')

score_fu_a1c_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q1.csv')
score_fu_a1c_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q2.csv')
score_fu_a1c_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q3.csv')
score_fu_a1c_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q4.csv')
score_fu_a1c_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q5.csv')
score_fu_a1c_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q6.csv')
score_fu_a1c_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q7.csv')
score_fu_a1c_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q8.csv')
score_fu_a1c_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q9.csv')
score_fu_a1c_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_elmo_top10_Q10.csv')

score_fu_noLab_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q1.csv')
score_fu_noLab_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q2.csv')
score_fu_noLab_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q3.csv')
score_fu_noLab_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q4.csv')
score_fu_noLab_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q5.csv')
score_fu_noLab_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q6.csv')
score_fu_noLab_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q7.csv')
score_fu_noLab_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q8.csv')
score_fu_noLab_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q9.csv')
score_fu_noLab_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_elmo_top10_Q10.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

score_zou_cre_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q1.csv')
score_zou_cre_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q2.csv')
score_zou_cre_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q3.csv')
score_zou_cre_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q4.csv')
score_zou_cre_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q5.csv')
score_zou_cre_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q6.csv')
score_zou_cre_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q7.csv')
score_zou_cre_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q8.csv')
score_zou_cre_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q9.csv')
score_zou_cre_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_elmo_top10_Q10.csv')

score_zou_glu_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q1.csv')
score_zou_glu_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q2.csv')
score_zou_glu_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q3.csv')
score_zou_glu_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q4.csv')
score_zou_glu_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q5.csv')
score_zou_glu_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q6.csv')
score_zou_glu_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q7.csv')
score_zou_glu_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q8.csv')
score_zou_glu_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q9.csv')
score_zou_glu_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_elmo_top10_Q10.csv')

score_zou_a1c_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q1.csv')
score_zou_a1c_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q2.csv')
score_zou_a1c_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q3.csv')
score_zou_a1c_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q4.csv')
score_zou_a1c_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q5.csv')
score_zou_a1c_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q6.csv')
score_zou_a1c_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q7.csv')
score_zou_a1c_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q8.csv')
score_zou_a1c_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q9.csv')
score_zou_a1c_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_elmo_top10_Q10.csv')

score_zou_noLab_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q1.csv')
score_zou_noLab_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q2.csv')
score_zou_noLab_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q3.csv')
score_zou_noLab_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q4.csv')
score_zou_noLab_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q5.csv')
score_zou_noLab_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q6.csv')
score_zou_noLab_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q7.csv')
score_zou_noLab_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q8.csv')
score_zou_noLab_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q9.csv')
score_zou_noLab_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_elmo_top10_Q10.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

score_zhang_cre_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q1.csv')
score_zhang_cre_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q2.csv')
score_zhang_cre_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q3.csv')
score_zhang_cre_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q4.csv')
score_zhang_cre_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q5.csv')
score_zhang_cre_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q6.csv')
score_zhang_cre_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q7.csv')
score_zhang_cre_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q8.csv')
score_zhang_cre_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q9.csv')
score_zhang_cre_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_elmo_top10_Q10.csv')

score_zhang_glu_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q1.csv')
score_zhang_glu_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q2.csv')
score_zhang_glu_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q3.csv')
score_zhang_glu_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q4.csv')
score_zhang_glu_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q5.csv')
score_zhang_glu_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q6.csv')
score_zhang_glu_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q7.csv')
score_zhang_glu_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q8.csv')
score_zhang_glu_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q9.csv')
score_zhang_glu_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_elmo_top10_Q10.csv')

score_zhang_a1c_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q1.csv')
score_zhang_a1c_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q2.csv')
score_zhang_a1c_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q3.csv')
score_zhang_a1c_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q4.csv')
score_zhang_a1c_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q5.csv')
score_zhang_a1c_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q6.csv')
score_zhang_a1c_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q7.csv')
score_zhang_a1c_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q8.csv')
score_zhang_a1c_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q9.csv')
score_zhang_a1c_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_elmo_top10_Q10.csv')

score_zhang_noLab_elmo_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q1.csv')
score_zhang_noLab_elmo_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q2.csv')
score_zhang_noLab_elmo_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q3.csv')
score_zhang_noLab_elmo_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q4.csv')
score_zhang_noLab_elmo_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q5.csv')
score_zhang_noLab_elmo_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q6.csv')
score_zhang_noLab_elmo_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q7.csv')
score_zhang_noLab_elmo_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q8.csv')
score_zhang_noLab_elmo_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q9.csv')
score_zhang_noLab_elmo_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_elmo_top10_Q10.csv')

# extract similarity scores for lab_tfidf
score_fu_cre_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q1.csv')
score_fu_cre_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q2.csv')
score_fu_cre_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q3.csv')
score_fu_cre_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q4.csv')
score_fu_cre_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q5.csv')
score_fu_cre_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q6.csv')
score_fu_cre_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q7.csv')
score_fu_cre_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q8.csv')
score_fu_cre_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q9.csv')
score_fu_cre_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Creatinine_tfidf_top10_Q10.csv')

score_fu_glu_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q1.csv')
score_fu_glu_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q2.csv')
score_fu_glu_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q3.csv')
score_fu_glu_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q4.csv')
score_fu_glu_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q5.csv')
score_fu_glu_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q6.csv')
score_fu_glu_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q7.csv')
score_fu_glu_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q8.csv')
score_fu_glu_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q9.csv')
score_fu_glu_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/Glucose_tfidf_top10_Q10.csv')

score_fu_a1c_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q1.csv')
score_fu_a1c_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q2.csv')
score_fu_a1c_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q3.csv')
score_fu_a1c_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q4.csv')
score_fu_a1c_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q5.csv')
score_fu_a1c_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q6.csv')
score_fu_a1c_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q7.csv')
score_fu_a1c_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q8.csv')
score_fu_a1c_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q9.csv')
score_fu_a1c_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/HbA1c_tfidf_top10_Q10.csv')

score_fu_noLab_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q1.csv')
score_fu_noLab_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q2.csv')
score_fu_noLab_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q3.csv')
score_fu_noLab_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q4.csv')
score_fu_noLab_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q5.csv')
score_fu_noLab_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q6.csv')
score_fu_noLab_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q7.csv')
score_fu_noLab_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q8.csv')
score_fu_noLab_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q9.csv')
score_fu_noLab_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Fanzhe/NoLab_tfidf_top10_Q10.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

score_zou_cre_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q1.csv')
score_zou_cre_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q2.csv')
score_zou_cre_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q3.csv')
score_zou_cre_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q4.csv')
score_zou_cre_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q5.csv')
score_zou_cre_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q6.csv')
score_zou_cre_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q7.csv')
score_zou_cre_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q8.csv')
score_zou_cre_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q9.csv')
score_zou_cre_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Creatinine_tfidf_top10_Q10.csv')

score_zou_glu_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q1.csv')
score_zou_glu_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q2.csv')
score_zou_glu_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q3.csv')
score_zou_glu_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q4.csv')
score_zou_glu_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q5.csv')
score_zou_glu_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q6.csv')
score_zou_glu_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q7.csv')
score_zou_glu_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q8.csv')
score_zou_glu_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q9.csv')
score_zou_glu_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/Glucose_tfidf_top10_Q10.csv')

score_zou_a1c_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q1.csv')
score_zou_a1c_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q2.csv')
score_zou_a1c_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q3.csv')
score_zou_a1c_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q4.csv')
score_zou_a1c_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q5.csv')
score_zou_a1c_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q6.csv')
score_zou_a1c_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q7.csv')
score_zou_a1c_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q8.csv')
score_zou_a1c_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q9.csv')
score_zou_a1c_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/HbA1c_tfidf_top10_Q10.csv')

score_zou_noLab_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q1.csv')
score_zou_noLab_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q2.csv')
score_zou_noLab_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q3.csv')
score_zou_noLab_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q4.csv')
score_zou_noLab_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q5.csv')
score_zou_noLab_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q6.csv')
score_zou_noLab_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q7.csv')
score_zou_noLab_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q8.csv')
score_zou_noLab_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q9.csv')
score_zou_noLab_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Guannan/NoLab_tfidf_top10_Q10.csv')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

score_zhang_cre_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q1.csv')
score_zhang_cre_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q2.csv')
score_zhang_cre_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q3.csv')
score_zhang_cre_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q4.csv')
score_zhang_cre_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q5.csv')
score_zhang_cre_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q6.csv')
score_zhang_cre_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q7.csv')
score_zhang_cre_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q8.csv')
score_zhang_cre_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q9.csv')
score_zhang_cre_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Creatinine_tfidf_top10_Q10.csv')

score_zhang_glu_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q1.csv')
score_zhang_glu_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q2.csv')
score_zhang_glu_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q3.csv')
score_zhang_glu_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q4.csv')
score_zhang_glu_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q5.csv')
score_zhang_glu_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q6.csv')
score_zhang_glu_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q7.csv')
score_zhang_glu_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q8.csv')
score_zhang_glu_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q9.csv')
score_zhang_glu_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/Glucose_tfidf_top10_Q10.csv')

score_zhang_a1c_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q1.csv')
score_zhang_a1c_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q2.csv')
score_zhang_a1c_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q3.csv')
score_zhang_a1c_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q4.csv')
score_zhang_a1c_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q5.csv')
score_zhang_a1c_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q6.csv')
score_zhang_a1c_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q7.csv')
score_zhang_a1c_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q8.csv')
score_zhang_a1c_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q9.csv')
score_zhang_a1c_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/HbA1c_tfidf_top10_Q10.csv')

score_zhang_noLab_tfidf_Q1 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q1.csv')
score_zhang_noLab_tfidf_Q2 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q2.csv')
score_zhang_noLab_tfidf_Q3 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q3.csv')
score_zhang_noLab_tfidf_Q4 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q4.csv')
score_zhang_noLab_tfidf_Q5 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q5.csv')
score_zhang_noLab_tfidf_Q6 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q6.csv')
score_zhang_noLab_tfidf_Q7 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q7.csv')
score_zhang_noLab_tfidf_Q8 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q8.csv')
score_zhang_noLab_tfidf_Q9 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q9.csv')
score_zhang_noLab_tfidf_Q10 = load_score('/Users/luyu/Documents/Master Thesis/rating/csv_Zhan/NoLab_tfidf_top10_Q10.csv')

#print(score_fu_cre_allModels_Q1)
#print(score_zou_cre_allModels_Q1)
#print(score_zhang_cre_allModels_Q1)

# calculate gold standard for lab_allModels_top10
gs_cre_allModels_Q1 = get_gold_standard(score_fu_cre_allModels_Q1, score_zou_cre_allModels_Q1, score_zhang_cre_allModels_Q1)
gs_cre_allModels_Q2 = get_gold_standard(score_fu_cre_allModels_Q2, score_zou_cre_allModels_Q2, score_zhang_cre_allModels_Q2)
gs_cre_allModels_Q3 = get_gold_standard(score_fu_cre_allModels_Q3, score_zou_cre_allModels_Q3, score_zhang_cre_allModels_Q3)
gs_cre_allModels_Q4 = get_gold_standard(score_fu_cre_allModels_Q4, score_zou_cre_allModels_Q4, score_zhang_cre_allModels_Q4)
gs_cre_allModels_Q5 = get_gold_standard(score_fu_cre_allModels_Q5, score_zou_cre_allModels_Q5, score_zhang_cre_allModels_Q5)
gs_cre_allModels_Q6 = get_gold_standard(score_fu_cre_allModels_Q6, score_zou_cre_allModels_Q6, score_zhang_cre_allModels_Q6)
gs_cre_allModels_Q7 = get_gold_standard(score_fu_cre_allModels_Q7, score_zou_cre_allModels_Q7, score_zhang_cre_allModels_Q7)
gs_cre_allModels_Q8 = get_gold_standard(score_fu_cre_allModels_Q8, score_zou_cre_allModels_Q8, score_zhang_cre_allModels_Q8)
gs_cre_allModels_Q9 = get_gold_standard(score_fu_cre_allModels_Q9, score_zou_cre_allModels_Q9, score_zhang_cre_allModels_Q9)
gs_cre_allModels_Q10 = get_gold_standard(score_fu_cre_allModels_Q10, score_zou_cre_allModels_Q10, score_zhang_cre_allModels_Q10)

gs_glu_allModels_Q1 = get_gold_standard(score_fu_glu_allModels_Q1, score_zou_glu_allModels_Q1, score_zhang_glu_allModels_Q1)
gs_glu_allModels_Q2 = get_gold_standard(score_fu_glu_allModels_Q2, score_zou_glu_allModels_Q2, score_zhang_glu_allModels_Q2)
gs_glu_allModels_Q3 = get_gold_standard(score_fu_glu_allModels_Q3, score_zou_glu_allModels_Q3, score_zhang_glu_allModels_Q3)
gs_glu_allModels_Q4 = get_gold_standard(score_fu_glu_allModels_Q4, score_zou_glu_allModels_Q4, score_zhang_glu_allModels_Q4)
gs_glu_allModels_Q5 = get_gold_standard(score_fu_glu_allModels_Q5, score_zou_glu_allModels_Q5, score_zhang_glu_allModels_Q5)
gs_glu_allModels_Q6 = get_gold_standard(score_fu_glu_allModels_Q6, score_zou_glu_allModels_Q6, score_zhang_glu_allModels_Q6)
gs_glu_allModels_Q7 = get_gold_standard(score_fu_glu_allModels_Q7, score_zou_glu_allModels_Q7, score_zhang_glu_allModels_Q7)
gs_glu_allModels_Q8 = get_gold_standard(score_fu_glu_allModels_Q8, score_zou_glu_allModels_Q8, score_zhang_glu_allModels_Q8)
gs_glu_allModels_Q9 = get_gold_standard(score_fu_glu_allModels_Q9, score_zou_glu_allModels_Q9, score_zhang_glu_allModels_Q9)
gs_glu_allModels_Q10 = get_gold_standard(score_fu_glu_allModels_Q10, score_zou_glu_allModels_Q10, score_zhang_glu_allModels_Q10)

gs_a1c_allModels_Q1 = get_gold_standard(score_fu_a1c_allModels_Q1, score_zou_a1c_allModels_Q1, score_zhang_a1c_allModels_Q1)
gs_a1c_allModels_Q2 = get_gold_standard(score_fu_a1c_allModels_Q2, score_zou_a1c_allModels_Q2, score_zhang_a1c_allModels_Q2)
gs_a1c_allModels_Q3 = get_gold_standard(score_fu_a1c_allModels_Q3, score_zou_a1c_allModels_Q3, score_zhang_a1c_allModels_Q3)
gs_a1c_allModels_Q4 = get_gold_standard(score_fu_a1c_allModels_Q4, score_zou_a1c_allModels_Q4, score_zhang_a1c_allModels_Q4)
gs_a1c_allModels_Q5 = get_gold_standard(score_fu_a1c_allModels_Q5, score_zou_a1c_allModels_Q5, score_zhang_a1c_allModels_Q5)
gs_a1c_allModels_Q6 = get_gold_standard(score_fu_a1c_allModels_Q6, score_zou_a1c_allModels_Q6, score_zhang_a1c_allModels_Q6)
gs_a1c_allModels_Q7 = get_gold_standard(score_fu_a1c_allModels_Q7, score_zou_a1c_allModels_Q7, score_zhang_a1c_allModels_Q7)
gs_a1c_allModels_Q8 = get_gold_standard(score_fu_a1c_allModels_Q8, score_zou_a1c_allModels_Q8, score_zhang_a1c_allModels_Q8)
gs_a1c_allModels_Q9 = get_gold_standard(score_fu_a1c_allModels_Q9, score_zou_a1c_allModels_Q9, score_zhang_a1c_allModels_Q9)
gs_a1c_allModels_Q10 = get_gold_standard(score_fu_a1c_allModels_Q10, score_zou_a1c_allModels_Q10, score_zhang_a1c_allModels_Q10)

gs_noLab_allModels_Q1 = get_gold_standard(score_fu_noLab_allModels_Q1, score_zou_noLab_allModels_Q1, score_zhang_noLab_allModels_Q1)
gs_noLab_allModels_Q2 = get_gold_standard(score_fu_noLab_allModels_Q2, score_zou_noLab_allModels_Q2, score_zhang_noLab_allModels_Q2)
gs_noLab_allModels_Q3 = get_gold_standard(score_fu_noLab_allModels_Q3, score_zou_noLab_allModels_Q3, score_zhang_noLab_allModels_Q3)
gs_noLab_allModels_Q4 = get_gold_standard(score_fu_noLab_allModels_Q4, score_zou_noLab_allModels_Q4, score_zhang_noLab_allModels_Q4)
gs_noLab_allModels_Q5 = get_gold_standard(score_fu_noLab_allModels_Q5, score_zou_noLab_allModels_Q5, score_zhang_noLab_allModels_Q5)
gs_noLab_allModels_Q6 = get_gold_standard(score_fu_noLab_allModels_Q6, score_zou_noLab_allModels_Q6, score_zhang_noLab_allModels_Q6)
gs_noLab_allModels_Q7 = get_gold_standard(score_fu_noLab_allModels_Q7, score_zou_noLab_allModels_Q7, score_zhang_noLab_allModels_Q7)
gs_noLab_allModels_Q8 = get_gold_standard(score_fu_noLab_allModels_Q8, score_zou_noLab_allModels_Q8, score_zhang_noLab_allModels_Q8)
gs_noLab_allModels_Q9 = get_gold_standard(score_fu_noLab_allModels_Q9, score_zou_noLab_allModels_Q9, score_zhang_noLab_allModels_Q9)
gs_noLab_allModels_Q10 = get_gold_standard(score_fu_noLab_allModels_Q10, score_zou_noLab_allModels_Q10, score_zhang_noLab_allModels_Q10)

# calculate gold standard for lab_ELMo_top10
gs_cre_elmo_Q1 = get_gold_standard(score_fu_cre_elmo_Q1, score_zou_cre_elmo_Q1, score_zhang_cre_elmo_Q1)
gs_cre_elmo_Q2 = get_gold_standard(score_fu_cre_elmo_Q2, score_zou_cre_elmo_Q2, score_zhang_cre_elmo_Q2)
gs_cre_elmo_Q3 = get_gold_standard(score_fu_cre_elmo_Q3, score_zou_cre_elmo_Q3, score_zhang_cre_elmo_Q3)
gs_cre_elmo_Q4 = get_gold_standard(score_fu_cre_elmo_Q4, score_zou_cre_elmo_Q4, score_zhang_cre_elmo_Q4)
gs_cre_elmo_Q5 = get_gold_standard(score_fu_cre_elmo_Q5, score_zou_cre_elmo_Q5, score_zhang_cre_elmo_Q5)
gs_cre_elmo_Q6 = get_gold_standard(score_fu_cre_elmo_Q6, score_zou_cre_elmo_Q6, score_zhang_cre_elmo_Q6)
gs_cre_elmo_Q7 = get_gold_standard(score_fu_cre_elmo_Q7, score_zou_cre_elmo_Q7, score_zhang_cre_elmo_Q7)
gs_cre_elmo_Q8 = get_gold_standard(score_fu_cre_elmo_Q8, score_zou_cre_elmo_Q8, score_zhang_cre_elmo_Q8)
gs_cre_elmo_Q9 = get_gold_standard(score_fu_cre_elmo_Q9, score_zou_cre_elmo_Q9, score_zhang_cre_elmo_Q9)
gs_cre_elmo_Q10 = get_gold_standard(score_fu_cre_elmo_Q10, score_zou_cre_elmo_Q10, score_zhang_cre_elmo_Q10)

gs_glu_elmo_Q1 = get_gold_standard(score_fu_glu_elmo_Q1, score_zou_glu_elmo_Q1, score_zhang_glu_elmo_Q1)
gs_glu_elmo_Q2 = get_gold_standard(score_fu_glu_elmo_Q2, score_zou_glu_elmo_Q2, score_zhang_glu_elmo_Q2)
gs_glu_elmo_Q3 = get_gold_standard(score_fu_glu_elmo_Q3, score_zou_glu_elmo_Q3, score_zhang_glu_elmo_Q3)
gs_glu_elmo_Q4 = get_gold_standard(score_fu_glu_elmo_Q4, score_zou_glu_elmo_Q4, score_zhang_glu_elmo_Q4)
gs_glu_elmo_Q5 = get_gold_standard(score_fu_glu_elmo_Q5, score_zou_glu_elmo_Q5, score_zhang_glu_elmo_Q5)
gs_glu_elmo_Q6 = get_gold_standard(score_fu_glu_elmo_Q6, score_zou_glu_elmo_Q6, score_zhang_glu_elmo_Q6)
gs_glu_elmo_Q7 = get_gold_standard(score_fu_glu_elmo_Q7, score_zou_glu_elmo_Q7, score_zhang_glu_elmo_Q7)
gs_glu_elmo_Q8 = get_gold_standard(score_fu_glu_elmo_Q8, score_zou_glu_elmo_Q8, score_zhang_glu_elmo_Q8)
gs_glu_elmo_Q9 = get_gold_standard(score_fu_glu_elmo_Q9, score_zou_glu_elmo_Q9, score_zhang_glu_elmo_Q9)
gs_glu_elmo_Q10 = get_gold_standard(score_fu_glu_elmo_Q10, score_zou_glu_elmo_Q10, score_zhang_glu_elmo_Q10)

gs_a1c_elmo_Q1 = get_gold_standard(score_fu_a1c_elmo_Q1, score_zou_a1c_elmo_Q1, score_zhang_a1c_elmo_Q1)
gs_a1c_elmo_Q2 = get_gold_standard(score_fu_a1c_elmo_Q2, score_zou_a1c_elmo_Q2, score_zhang_a1c_elmo_Q2)
gs_a1c_elmo_Q3 = get_gold_standard(score_fu_a1c_elmo_Q3, score_zou_a1c_elmo_Q3, score_zhang_a1c_elmo_Q3)
gs_a1c_elmo_Q4 = get_gold_standard(score_fu_a1c_elmo_Q4, score_zou_a1c_elmo_Q4, score_zhang_a1c_elmo_Q4)
gs_a1c_elmo_Q5 = get_gold_standard(score_fu_a1c_elmo_Q5, score_zou_a1c_elmo_Q5, score_zhang_a1c_elmo_Q5)
gs_a1c_elmo_Q6 = get_gold_standard(score_fu_a1c_elmo_Q6, score_zou_a1c_elmo_Q6, score_zhang_a1c_elmo_Q6)
gs_a1c_elmo_Q7 = get_gold_standard(score_fu_a1c_elmo_Q7, score_zou_a1c_elmo_Q7, score_zhang_a1c_elmo_Q7)
gs_a1c_elmo_Q8 = get_gold_standard(score_fu_a1c_elmo_Q8, score_zou_a1c_elmo_Q8, score_zhang_a1c_elmo_Q8)
gs_a1c_elmo_Q9 = get_gold_standard(score_fu_a1c_elmo_Q9, score_zou_a1c_elmo_Q9, score_zhang_a1c_elmo_Q9)
gs_a1c_elmo_Q10 = get_gold_standard(score_fu_a1c_elmo_Q10, score_zou_a1c_elmo_Q10, score_zhang_a1c_elmo_Q10)

gs_noLab_elmo_Q1 = get_gold_standard(score_fu_noLab_elmo_Q1, score_zou_noLab_elmo_Q1, score_zhang_noLab_elmo_Q1)
gs_noLab_elmo_Q2 = get_gold_standard(score_fu_noLab_elmo_Q2, score_zou_noLab_elmo_Q2, score_zhang_noLab_elmo_Q2)
gs_noLab_elmo_Q3 = get_gold_standard(score_fu_noLab_elmo_Q3, score_zou_noLab_elmo_Q3, score_zhang_noLab_elmo_Q3)
gs_noLab_elmo_Q4 = get_gold_standard(score_fu_noLab_elmo_Q4, score_zou_noLab_elmo_Q4, score_zhang_noLab_elmo_Q4)
gs_noLab_elmo_Q5 = get_gold_standard(score_fu_noLab_elmo_Q5, score_zou_noLab_elmo_Q5, score_zhang_noLab_elmo_Q5)
gs_noLab_elmo_Q6 = get_gold_standard(score_fu_noLab_elmo_Q6, score_zou_noLab_elmo_Q6, score_zhang_noLab_elmo_Q6)
gs_noLab_elmo_Q7 = get_gold_standard(score_fu_noLab_elmo_Q7, score_zou_noLab_elmo_Q7, score_zhang_noLab_elmo_Q7)
gs_noLab_elmo_Q8 = get_gold_standard(score_fu_noLab_elmo_Q8, score_zou_noLab_elmo_Q8, score_zhang_noLab_elmo_Q8)
gs_noLab_elmo_Q9 = get_gold_standard(score_fu_noLab_elmo_Q9, score_zou_noLab_elmo_Q9, score_zhang_noLab_elmo_Q9)
gs_noLab_elmo_Q10 = get_gold_standard(score_fu_noLab_elmo_Q10, score_zou_noLab_elmo_Q10, score_zhang_noLab_elmo_Q10)

# calculate gold standard for lab_tfidf_top10
gs_cre_tfidf_Q1 = get_gold_standard(score_fu_cre_tfidf_Q1, score_zou_cre_tfidf_Q1, score_zhang_cre_tfidf_Q1)
gs_cre_tfidf_Q2 = get_gold_standard(score_fu_cre_tfidf_Q2, score_zou_cre_tfidf_Q2, score_zhang_cre_tfidf_Q2)
gs_cre_tfidf_Q3 = get_gold_standard(score_fu_cre_tfidf_Q3, score_zou_cre_tfidf_Q3, score_zhang_cre_tfidf_Q3)
gs_cre_tfidf_Q4 = get_gold_standard(score_fu_cre_tfidf_Q4, score_zou_cre_tfidf_Q4, score_zhang_cre_tfidf_Q4)
gs_cre_tfidf_Q5 = get_gold_standard(score_fu_cre_tfidf_Q5, score_zou_cre_tfidf_Q5, score_zhang_cre_tfidf_Q5)
gs_cre_tfidf_Q6 = get_gold_standard(score_fu_cre_tfidf_Q6, score_zou_cre_tfidf_Q6, score_zhang_cre_tfidf_Q6)
gs_cre_tfidf_Q7 = get_gold_standard(score_fu_cre_tfidf_Q7, score_zou_cre_tfidf_Q7, score_zhang_cre_tfidf_Q7)
gs_cre_tfidf_Q8 = get_gold_standard(score_fu_cre_tfidf_Q8, score_zou_cre_tfidf_Q8, score_zhang_cre_tfidf_Q8)
gs_cre_tfidf_Q9 = get_gold_standard(score_fu_cre_tfidf_Q9, score_zou_cre_tfidf_Q9, score_zhang_cre_tfidf_Q9)
gs_cre_tfidf_Q10 = get_gold_standard(score_fu_cre_tfidf_Q10, score_zou_cre_tfidf_Q10, score_zhang_cre_tfidf_Q10)

gs_glu_tfidf_Q1 = get_gold_standard(score_fu_glu_tfidf_Q1, score_zou_glu_tfidf_Q1, score_zhang_glu_tfidf_Q1)
gs_glu_tfidf_Q2 = get_gold_standard(score_fu_glu_tfidf_Q2, score_zou_glu_tfidf_Q2, score_zhang_glu_tfidf_Q2)
gs_glu_tfidf_Q3 = get_gold_standard(score_fu_glu_tfidf_Q3, score_zou_glu_tfidf_Q3, score_zhang_glu_tfidf_Q3)
gs_glu_tfidf_Q4 = get_gold_standard(score_fu_glu_tfidf_Q4, score_zou_glu_tfidf_Q4, score_zhang_glu_tfidf_Q4)
gs_glu_tfidf_Q5 = get_gold_standard(score_fu_glu_tfidf_Q5, score_zou_glu_tfidf_Q5, score_zhang_glu_tfidf_Q5)
gs_glu_tfidf_Q6 = get_gold_standard(score_fu_glu_tfidf_Q6, score_zou_glu_tfidf_Q6, score_zhang_glu_tfidf_Q6)
gs_glu_tfidf_Q7 = get_gold_standard(score_fu_glu_tfidf_Q7, score_zou_glu_tfidf_Q7, score_zhang_glu_tfidf_Q7)
gs_glu_tfidf_Q8 = get_gold_standard(score_fu_glu_tfidf_Q8, score_zou_glu_tfidf_Q8, score_zhang_glu_tfidf_Q8)
gs_glu_tfidf_Q9 = get_gold_standard(score_fu_glu_tfidf_Q9, score_zou_glu_tfidf_Q9, score_zhang_glu_tfidf_Q9)
gs_glu_tfidf_Q10 = get_gold_standard(score_fu_glu_tfidf_Q10, score_zou_glu_tfidf_Q10, score_zhang_glu_tfidf_Q10)

gs_a1c_tfidf_Q1 = get_gold_standard(score_fu_a1c_tfidf_Q1, score_zou_a1c_tfidf_Q1, score_zhang_a1c_tfidf_Q1)
gs_a1c_tfidf_Q2 = get_gold_standard(score_fu_a1c_tfidf_Q2, score_zou_a1c_tfidf_Q2, score_zhang_a1c_tfidf_Q2)
gs_a1c_tfidf_Q3 = get_gold_standard(score_fu_a1c_tfidf_Q3, score_zou_a1c_tfidf_Q3, score_zhang_a1c_tfidf_Q3)
gs_a1c_tfidf_Q4 = get_gold_standard(score_fu_a1c_tfidf_Q4, score_zou_a1c_tfidf_Q4, score_zhang_a1c_tfidf_Q4)
gs_a1c_tfidf_Q5 = get_gold_standard(score_fu_a1c_tfidf_Q5, score_zou_a1c_tfidf_Q5, score_zhang_a1c_tfidf_Q5)
gs_a1c_tfidf_Q6 = get_gold_standard(score_fu_a1c_tfidf_Q6, score_zou_a1c_tfidf_Q6, score_zhang_a1c_tfidf_Q6)
gs_a1c_tfidf_Q7 = get_gold_standard(score_fu_a1c_tfidf_Q7, score_zou_a1c_tfidf_Q7, score_zhang_a1c_tfidf_Q7)
gs_a1c_tfidf_Q8 = get_gold_standard(score_fu_a1c_tfidf_Q8, score_zou_a1c_tfidf_Q8, score_zhang_a1c_tfidf_Q8)
gs_a1c_tfidf_Q9 = get_gold_standard(score_fu_a1c_tfidf_Q9, score_zou_a1c_tfidf_Q9, score_zhang_a1c_tfidf_Q9)
gs_a1c_tfidf_Q10 = get_gold_standard(score_fu_a1c_tfidf_Q10, score_zou_a1c_tfidf_Q10, score_zhang_a1c_tfidf_Q10)

gs_noLab_tfidf_Q1 = get_gold_standard(score_fu_noLab_tfidf_Q1, score_zou_noLab_tfidf_Q1, score_zhang_noLab_tfidf_Q1)
gs_noLab_tfidf_Q2 = get_gold_standard(score_fu_noLab_tfidf_Q2, score_zou_noLab_tfidf_Q2, score_zhang_noLab_tfidf_Q2)
gs_noLab_tfidf_Q3 = get_gold_standard(score_fu_noLab_tfidf_Q3, score_zou_noLab_tfidf_Q3, score_zhang_noLab_tfidf_Q3)
gs_noLab_tfidf_Q4 = get_gold_standard(score_fu_noLab_tfidf_Q4, score_zou_noLab_tfidf_Q4, score_zhang_noLab_tfidf_Q4)
gs_noLab_tfidf_Q5 = get_gold_standard(score_fu_noLab_tfidf_Q5, score_zou_noLab_tfidf_Q5, score_zhang_noLab_tfidf_Q5)
gs_noLab_tfidf_Q6 = get_gold_standard(score_fu_noLab_tfidf_Q6, score_zou_noLab_tfidf_Q6, score_zhang_noLab_tfidf_Q6)
gs_noLab_tfidf_Q7 = get_gold_standard(score_fu_noLab_tfidf_Q7, score_zou_noLab_tfidf_Q7, score_zhang_noLab_tfidf_Q7)
gs_noLab_tfidf_Q8 = get_gold_standard(score_fu_noLab_tfidf_Q8, score_zou_noLab_tfidf_Q8, score_zhang_noLab_tfidf_Q8)
gs_noLab_tfidf_Q9 = get_gold_standard(score_fu_noLab_tfidf_Q9, score_zou_noLab_tfidf_Q9, score_zhang_noLab_tfidf_Q9)
gs_noLab_tfidf_Q10 = get_gold_standard(score_fu_noLab_tfidf_Q10, score_zou_noLab_tfidf_Q10, score_zhang_noLab_tfidf_Q10)

#print(gs_cre_allModels_Q1)
#print(gs_cre_elmo_Q1)
#print(gs_cre_tfidf_Q1)

#print(gs_cre_allModels_Q2)
#print(gs_cre_elmo_Q2)
#print(gs_cre_tfidf_Q2)

#bag = zip(gs_a1c_allModels_Q1, gs_a1c_allModels_Q2, gs_a1c_allModels_Q3, gs_a1c_allModels_Q4, gs_a1c_allModels_Q5,
#          gs_a1c_allModels_Q6, gs_a1c_allModels_Q7, gs_a1c_allModels_Q8, gs_a1c_allModels_Q9, gs_a1c_allModels_Q10)
#bag = zip(gs_a1c_elmo_Q1, gs_a1c_elmo_Q2, gs_a1c_elmo_Q3, gs_a1c_elmo_Q4, gs_a1c_elmo_Q5,
#          gs_a1c_elmo_Q6, gs_a1c_elmo_Q7, gs_a1c_elmo_Q8, gs_a1c_elmo_Q9, gs_a1c_elmo_Q10)
#bag = zip(gs_a1c_tfidf_Q1, gs_a1c_tfidf_Q2, gs_a1c_tfidf_Q3, gs_a1c_tfidf_Q4, gs_a1c_tfidf_Q5,
#          gs_a1c_tfidf_Q6, gs_a1c_tfidf_Q7, gs_a1c_tfidf_Q8, gs_a1c_tfidf_Q9, gs_a1c_tfidf_Q10)
#bag = zip(gs_noLab_allModels_Q1, gs_noLab_elmo_Q1, gs_noLab_tfidf_Q1)
#bag = zip(gs_a1c_allModels_Q8, gs_a1c_elmo_Q8, gs_a1c_tfidf_Q8)
#bag = zip(gs_a1c_allModels_Q1, gs_a1c_elmo_Q1, gs_a1c_tfidf_Q1)
#bag = zip(gs_a1c_allModels_Q2, gs_a1c_elmo_Q2, gs_a1c_tfidf_Q2)
#bag = zip(gs_a1c_allModels_Q3, gs_a1c_elmo_Q3, gs_a1c_tfidf_Q3)
#bag = zip(gs_a1c_allModels_Q4, gs_a1c_elmo_Q4, gs_a1c_tfidf_Q4)
#bag = zip(gs_a1c_allModels_Q5, gs_a1c_elmo_Q5, gs_a1c_tfidf_Q5)
#bag = zip(gs_a1c_allModels_Q6, gs_a1c_elmo_Q6, gs_a1c_tfidf_Q6)
#bag = zip(gs_a1c_allModels_Q7, gs_a1c_elmo_Q7, gs_a1c_tfidf_Q7)
#bag = zip(gs_a1c_allModels_Q8, gs_a1c_elmo_Q8, gs_a1c_tfidf_Q8)
#bag = zip(gs_a1c_allModels_Q9, gs_a1c_elmo_Q9, gs_a1c_tfidf_Q9)
#bag = zip(gs_a1c_allModels_Q10, gs_a1c_elmo_Q10, gs_a1c_tfidf_Q10)

#bag = zip(gs_noLab_allModels_Q1, gs_noLab_elmo_Q1, gs_noLab_tfidf_Q1)
#bag = zip(gs_noLab_allModels_Q2, gs_noLab_elmo_Q2, gs_noLab_tfidf_Q2)
#bag = zip(gs_noLab_allModels_Q3, gs_noLab_elmo_Q3, gs_noLab_tfidf_Q3)
#bag = zip(gs_noLab_allModels_Q4, gs_noLab_elmo_Q4, gs_noLab_tfidf_Q4)
#bag = zip(gs_noLab_allModels_Q5, gs_noLab_elmo_Q5, gs_noLab_tfidf_Q5)
#bag = zip(gs_noLab_allModels_Q6, gs_noLab_elmo_Q6, gs_noLab_tfidf_Q6)
#bag = zip(gs_noLab_allModels_Q7, gs_noLab_elmo_Q7, gs_noLab_tfidf_Q7)
#bag = zip(gs_noLab_allModels_Q8, gs_noLab_elmo_Q8, gs_noLab_tfidf_Q8)
#bag = zip(gs_noLab_allModels_Q9, gs_noLab_elmo_Q9, gs_noLab_tfidf_Q9)
#bag = zip(gs_noLab_allModels_Q10, gs_noLab_elmo_Q10, gs_noLab_tfidf_Q10)

#bag = zip(gs_glu_allModels_Q1, gs_glu_elmo_Q1, gs_glu_tfidf_Q1)
#bag = zip(gs_glu_allModels_Q2, gs_glu_elmo_Q2, gs_glu_tfidf_Q2)
#bag = zip(gs_glu_allModels_Q3, gs_glu_elmo_Q3, gs_glu_tfidf_Q3)
#bag = zip(gs_glu_allModels_Q4, gs_glu_elmo_Q4, gs_glu_tfidf_Q4)
bag = zip(gs_glu_allModels_Q5, gs_glu_elmo_Q5, gs_glu_tfidf_Q5)
#bag = zip(gs_glu_allModels_Q6, gs_glu_elmo_Q6, gs_glu_tfidf_Q6)
#bag = zip(gs_glu_allModels_Q7, gs_glu_elmo_Q7, gs_glu_tfidf_Q7)
#bag = zip(gs_glu_allModels_Q8, gs_glu_elmo_Q8, gs_glu_tfidf_Q8)
#bag = zip(gs_glu_allModels_Q9, gs_glu_elmo_Q9, gs_glu_tfidf_Q9)
#bag = zip(gs_glu_allModels_Q10, gs_glu_elmo_Q10, gs_glu_tfidf_Q10)

#bag = zip(gs_cre_allModels_Q1, gs_cre_elmo_Q1, gs_cre_tfidf_Q1)
#bag = zip(gs_cre_allModels_Q2, gs_cre_elmo_Q2, gs_cre_tfidf_Q2)
#bag = zip(gs_cre_allModels_Q3, gs_cre_elmo_Q3, gs_cre_tfidf_Q3)
#bag = zip(gs_cre_allModels_Q4, gs_cre_elmo_Q4, gs_cre_tfidf_Q4)
#bag = zip(gs_cre_allModels_Q5, gs_cre_elmo_Q5, gs_cre_tfidf_Q5)
#bag = zip(gs_cre_allModels_Q6, gs_cre_elmo_Q6, gs_cre_tfidf_Q6)
#bag = zip(gs_cre_allModels_Q7, gs_cre_elmo_Q7, gs_cre_tfidf_Q7)
#bag = zip(gs_cre_allModels_Q8, gs_cre_elmo_Q8, gs_cre_tfidf_Q8)
#bag = zip(gs_cre_allModels_Q9, gs_cre_elmo_Q9, gs_cre_tfidf_Q9)
#bag = zip(gs_cre_allModels_Q10, gs_cre_elmo_Q10, gs_cre_tfidf_Q10)
gs = pd.DataFrame(bag, columns = ['GS allModel', 'GS elmo', 'GS tfidf'])
#gs = pd.DataFrame(bag, columns = ['noLab Q1 - total - GS', 'noLab Q1 - elmo - GS', 'noLab Q1 - tfidf - GS'])

#gs = pd.DataFrame(bag, columns = ['Q1 - GS', 'Q2 - GS', 'Q3 - GS', 'Q4 - GS', 'Q5 - GS', 'Q6 - GS', 'Q7 - GS', 'Q8 - GS', 'Q9 - GS', 'Q10 - GS'])
gs.to_csv('/Users/luyu/Documents/Master Thesis/rating/test.csv', index=False)

# calculate nDCG at 1-10
# allModels
ndcg_cre_allModels_Q1 = [ndcg_at_k(gs_cre_allModels_Q1, i) for i in range(1, len(gs_cre_allModels_Q1)+1)]
ndcg_cre_allModels_Q2 = [ndcg_at_k(gs_cre_allModels_Q2, i) for i in range(1, len(gs_cre_allModels_Q2)+1)]
ndcg_cre_allModels_Q3 = [ndcg_at_k(gs_cre_allModels_Q3, i) for i in range(1, len(gs_cre_allModels_Q3)+1)]
ndcg_cre_allModels_Q4 = [ndcg_at_k(gs_cre_allModels_Q4, i) for i in range(1, len(gs_cre_allModels_Q4)+1)]
ndcg_cre_allModels_Q5 = [ndcg_at_k(gs_cre_allModels_Q5, i) for i in range(1, len(gs_cre_allModels_Q5)+1)]
ndcg_cre_allModels_Q6 = [ndcg_at_k(gs_cre_allModels_Q6, i) for i in range(1, len(gs_cre_allModels_Q6)+1)]
ndcg_cre_allModels_Q7 = [ndcg_at_k(gs_cre_allModels_Q7, i) for i in range(1, len(gs_cre_allModels_Q7)+1)]
ndcg_cre_allModels_Q8 = [ndcg_at_k(gs_cre_allModels_Q8, i) for i in range(1, len(gs_cre_allModels_Q8)+1)]
ndcg_cre_allModels_Q9 = [ndcg_at_k(gs_cre_allModels_Q9, i) for i in range(1, len(gs_cre_allModels_Q9)+1)]
ndcg_cre_allModels_Q10 = [ndcg_at_k(gs_cre_allModels_Q10, i) for i in range(1, len(gs_cre_allModels_Q10)+1)]

ndcg_glu_allModels_Q1 = [ndcg_at_k(gs_glu_allModels_Q1, i) for i in range(1, len(gs_glu_allModels_Q1)+1)]
ndcg_glu_allModels_Q2 = [ndcg_at_k(gs_glu_allModels_Q2, i) for i in range(1, len(gs_glu_allModels_Q2)+1)]
ndcg_glu_allModels_Q3 = [ndcg_at_k(gs_glu_allModels_Q3, i) for i in range(1, len(gs_glu_allModels_Q3)+1)]
ndcg_glu_allModels_Q4 = [ndcg_at_k(gs_glu_allModels_Q4, i) for i in range(1, len(gs_glu_allModels_Q4)+1)]
ndcg_glu_allModels_Q5 = [ndcg_at_k(gs_glu_allModels_Q5, i) for i in range(1, len(gs_glu_allModels_Q5)+1)]
ndcg_glu_allModels_Q6 = [ndcg_at_k(gs_glu_allModels_Q6, i) for i in range(1, len(gs_glu_allModels_Q6)+1)]
ndcg_glu_allModels_Q7 = [ndcg_at_k(gs_glu_allModels_Q7, i) for i in range(1, len(gs_glu_allModels_Q7)+1)]
ndcg_glu_allModels_Q8 = [ndcg_at_k(gs_glu_allModels_Q8, i) for i in range(1, len(gs_glu_allModels_Q8)+1)]
ndcg_glu_allModels_Q9 = [ndcg_at_k(gs_glu_allModels_Q9, i) for i in range(1, len(gs_glu_allModels_Q9)+1)]
ndcg_glu_allModels_Q10 = [ndcg_at_k(gs_glu_allModels_Q10, i) for i in range(1, len(gs_glu_allModels_Q10)+1)]

ndcg_a1c_allModels_Q1 = [ndcg_at_k(gs_a1c_allModels_Q1, i) for i in range(1, len(gs_a1c_allModels_Q1)+1)]
ndcg_a1c_allModels_Q2 = [ndcg_at_k(gs_a1c_allModels_Q2, i) for i in range(1, len(gs_a1c_allModels_Q2)+1)]
ndcg_a1c_allModels_Q3 = [ndcg_at_k(gs_a1c_allModels_Q3, i) for i in range(1, len(gs_a1c_allModels_Q3)+1)]
ndcg_a1c_allModels_Q4 = [ndcg_at_k(gs_a1c_allModels_Q4, i) for i in range(1, len(gs_a1c_allModels_Q4)+1)]
ndcg_a1c_allModels_Q5 = [ndcg_at_k(gs_a1c_allModels_Q5, i) for i in range(1, len(gs_a1c_allModels_Q5)+1)]
ndcg_a1c_allModels_Q6 = [ndcg_at_k(gs_a1c_allModels_Q6, i) for i in range(1, len(gs_a1c_allModels_Q6)+1)]
ndcg_a1c_allModels_Q7 = [ndcg_at_k(gs_a1c_allModels_Q7, i) for i in range(1, len(gs_a1c_allModels_Q7)+1)]
ndcg_a1c_allModels_Q8 = [ndcg_at_k(gs_a1c_allModels_Q8, i) for i in range(1, len(gs_a1c_allModels_Q8)+1)]
ndcg_a1c_allModels_Q9 = [ndcg_at_k(gs_a1c_allModels_Q9, i) for i in range(1, len(gs_a1c_allModels_Q9)+1)]
ndcg_a1c_allModels_Q10 = [ndcg_at_k(gs_a1c_allModels_Q10, i) for i in range(1, len(gs_a1c_allModels_Q10)+1)]

# ELMo
ndcg_cre_elmo_Q1 = [ndcg_at_k(gs_cre_elmo_Q1, i) for i in range(1, len(gs_cre_elmo_Q1)+1)]
ndcg_cre_elmo_Q2 = [ndcg_at_k(gs_cre_elmo_Q2, i) for i in range(1, len(gs_cre_elmo_Q2)+1)]
ndcg_cre_elmo_Q3 = [ndcg_at_k(gs_cre_elmo_Q3, i) for i in range(1, len(gs_cre_elmo_Q3)+1)]
ndcg_cre_elmo_Q4 = [ndcg_at_k(gs_cre_elmo_Q4, i) for i in range(1, len(gs_cre_elmo_Q4)+1)]
ndcg_cre_elmo_Q5 = [ndcg_at_k(gs_cre_elmo_Q5, i) for i in range(1, len(gs_cre_elmo_Q5)+1)]
ndcg_cre_elmo_Q6 = [ndcg_at_k(gs_cre_elmo_Q6, i) for i in range(1, len(gs_cre_elmo_Q6)+1)]
ndcg_cre_elmo_Q7 = [ndcg_at_k(gs_cre_elmo_Q7, i) for i in range(1, len(gs_cre_elmo_Q7)+1)]
ndcg_cre_elmo_Q8 = [ndcg_at_k(gs_cre_elmo_Q8, i) for i in range(1, len(gs_cre_elmo_Q8)+1)]
ndcg_cre_elmo_Q9 = [ndcg_at_k(gs_cre_elmo_Q9, i) for i in range(1, len(gs_cre_elmo_Q9)+1)]
ndcg_cre_elmo_Q10 = [ndcg_at_k(gs_cre_elmo_Q10, i) for i in range(1, len(gs_cre_elmo_Q10)+1)]

ndcg_glu_elmo_Q1 = [ndcg_at_k(gs_glu_elmo_Q1, i) for i in range(1, len(gs_glu_elmo_Q1)+1)]
ndcg_glu_elmo_Q2 = [ndcg_at_k(gs_glu_elmo_Q2, i) for i in range(1, len(gs_glu_elmo_Q2)+1)]
ndcg_glu_elmo_Q3 = [ndcg_at_k(gs_glu_elmo_Q3, i) for i in range(1, len(gs_glu_elmo_Q3)+1)]
ndcg_glu_elmo_Q4 = [ndcg_at_k(gs_glu_elmo_Q4, i) for i in range(1, len(gs_glu_elmo_Q4)+1)]
ndcg_glu_elmo_Q5 = [ndcg_at_k(gs_glu_elmo_Q5, i) for i in range(1, len(gs_glu_elmo_Q5)+1)]
ndcg_glu_elmo_Q6 = [ndcg_at_k(gs_glu_elmo_Q6, i) for i in range(1, len(gs_glu_elmo_Q6)+1)]
ndcg_glu_elmo_Q7 = [ndcg_at_k(gs_glu_elmo_Q7, i) for i in range(1, len(gs_glu_elmo_Q7)+1)]
ndcg_glu_elmo_Q8 = [ndcg_at_k(gs_glu_elmo_Q8, i) for i in range(1, len(gs_glu_elmo_Q8)+1)]
ndcg_glu_elmo_Q9 = [ndcg_at_k(gs_glu_elmo_Q9, i) for i in range(1, len(gs_glu_elmo_Q9)+1)]
ndcg_glu_elmo_Q10 = [ndcg_at_k(gs_glu_elmo_Q10, i) for i in range(1, len(gs_glu_elmo_Q10)+1)]

ndcg_a1c_elmo_Q1 = [ndcg_at_k(gs_a1c_elmo_Q1, i) for i in range(1, len(gs_a1c_elmo_Q1)+1)]
ndcg_a1c_elmo_Q2 = [ndcg_at_k(gs_a1c_elmo_Q2, i) for i in range(1, len(gs_a1c_elmo_Q2)+1)]
ndcg_a1c_elmo_Q3 = [ndcg_at_k(gs_a1c_elmo_Q3, i) for i in range(1, len(gs_a1c_elmo_Q3)+1)]
ndcg_a1c_elmo_Q4 = [ndcg_at_k(gs_a1c_elmo_Q4, i) for i in range(1, len(gs_a1c_elmo_Q4)+1)]
ndcg_a1c_elmo_Q5 = [ndcg_at_k(gs_a1c_elmo_Q5, i) for i in range(1, len(gs_a1c_elmo_Q5)+1)]
ndcg_a1c_elmo_Q6 = [ndcg_at_k(gs_a1c_elmo_Q6, i) for i in range(1, len(gs_a1c_elmo_Q6)+1)]
ndcg_a1c_elmo_Q7 = [ndcg_at_k(gs_a1c_elmo_Q7, i) for i in range(1, len(gs_a1c_elmo_Q7)+1)]
ndcg_a1c_elmo_Q8 = [ndcg_at_k(gs_a1c_elmo_Q8, i) for i in range(1, len(gs_a1c_elmo_Q8)+1)]
ndcg_a1c_elmo_Q9 = [ndcg_at_k(gs_a1c_elmo_Q9, i) for i in range(1, len(gs_a1c_elmo_Q9)+1)]
ndcg_a1c_elmo_Q10 = [ndcg_at_k(gs_a1c_elmo_Q10, i) for i in range(1, len(gs_a1c_elmo_Q10)+1)]

# TF-IDF
ndcg_cre_tfidf_Q1 = [ndcg_at_k(gs_cre_tfidf_Q1, i) for i in range(1, len(gs_cre_tfidf_Q1)+1)]
ndcg_cre_tfidf_Q2 = [ndcg_at_k(gs_cre_tfidf_Q2, i) for i in range(1, len(gs_cre_tfidf_Q2)+1)]
ndcg_cre_tfidf_Q3 = [ndcg_at_k(gs_cre_tfidf_Q3, i) for i in range(1, len(gs_cre_tfidf_Q3)+1)]
ndcg_cre_tfidf_Q4 = [ndcg_at_k(gs_cre_tfidf_Q4, i) for i in range(1, len(gs_cre_tfidf_Q4)+1)]
ndcg_cre_tfidf_Q5 = [ndcg_at_k(gs_cre_tfidf_Q5, i) for i in range(1, len(gs_cre_tfidf_Q5)+1)]
ndcg_cre_tfidf_Q6 = [ndcg_at_k(gs_cre_tfidf_Q6, i) for i in range(1, len(gs_cre_tfidf_Q6)+1)]
ndcg_cre_tfidf_Q7 = [ndcg_at_k(gs_cre_tfidf_Q7, i) for i in range(1, len(gs_cre_tfidf_Q7)+1)]
ndcg_cre_tfidf_Q8 = [ndcg_at_k(gs_cre_tfidf_Q8, i) for i in range(1, len(gs_cre_tfidf_Q8)+1)]
ndcg_cre_tfidf_Q9 = [ndcg_at_k(gs_cre_tfidf_Q9, i) for i in range(1, len(gs_cre_tfidf_Q9)+1)]
ndcg_cre_tfidf_Q10 = [ndcg_at_k(gs_cre_tfidf_Q10, i) for i in range(1, len(gs_cre_tfidf_Q10)+1)]

ndcg_glu_tfidf_Q1 = [ndcg_at_k(gs_glu_tfidf_Q1, i) for i in range(1, len(gs_glu_tfidf_Q1)+1)]
ndcg_glu_tfidf_Q2 = [ndcg_at_k(gs_glu_tfidf_Q2, i) for i in range(1, len(gs_glu_tfidf_Q2)+1)]
ndcg_glu_tfidf_Q3 = [ndcg_at_k(gs_glu_tfidf_Q3, i) for i in range(1, len(gs_glu_tfidf_Q3)+1)]
ndcg_glu_tfidf_Q4 = [ndcg_at_k(gs_glu_tfidf_Q4, i) for i in range(1, len(gs_glu_tfidf_Q4)+1)]
ndcg_glu_tfidf_Q5 = [ndcg_at_k(gs_glu_tfidf_Q5, i) for i in range(1, len(gs_glu_tfidf_Q5)+1)]
ndcg_glu_tfidf_Q6 = [ndcg_at_k(gs_glu_tfidf_Q6, i) for i in range(1, len(gs_glu_tfidf_Q6)+1)]
ndcg_glu_tfidf_Q7 = [ndcg_at_k(gs_glu_tfidf_Q7, i) for i in range(1, len(gs_glu_tfidf_Q7)+1)]
ndcg_glu_tfidf_Q8 = [ndcg_at_k(gs_glu_tfidf_Q8, i) for i in range(1, len(gs_glu_tfidf_Q8)+1)]
ndcg_glu_tfidf_Q9 = [ndcg_at_k(gs_glu_tfidf_Q9, i) for i in range(1, len(gs_glu_tfidf_Q9)+1)]
ndcg_glu_tfidf_Q10 = [ndcg_at_k(gs_glu_tfidf_Q10, i) for i in range(1, len(gs_glu_tfidf_Q10)+1)]

ndcg_a1c_tfidf_Q1 = [ndcg_at_k(gs_a1c_tfidf_Q1, i) for i in range(1, len(gs_a1c_tfidf_Q1)+1)]
ndcg_a1c_tfidf_Q2 = [ndcg_at_k(gs_a1c_tfidf_Q2, i) for i in range(1, len(gs_a1c_tfidf_Q2)+1)]
ndcg_a1c_tfidf_Q3 = [ndcg_at_k(gs_a1c_tfidf_Q3, i) for i in range(1, len(gs_a1c_tfidf_Q3)+1)]
ndcg_a1c_tfidf_Q4 = [ndcg_at_k(gs_a1c_tfidf_Q4, i) for i in range(1, len(gs_a1c_tfidf_Q4)+1)]
ndcg_a1c_tfidf_Q5 = [ndcg_at_k(gs_a1c_tfidf_Q5, i) for i in range(1, len(gs_a1c_tfidf_Q5)+1)]
ndcg_a1c_tfidf_Q6 = [ndcg_at_k(gs_a1c_tfidf_Q6, i) for i in range(1, len(gs_a1c_tfidf_Q6)+1)]
ndcg_a1c_tfidf_Q7 = [ndcg_at_k(gs_a1c_tfidf_Q7, i) for i in range(1, len(gs_a1c_tfidf_Q7)+1)]
ndcg_a1c_tfidf_Q8 = [ndcg_at_k(gs_a1c_tfidf_Q8, i) for i in range(1, len(gs_a1c_tfidf_Q8)+1)]
ndcg_a1c_tfidf_Q9 = [ndcg_at_k(gs_a1c_tfidf_Q9, i) for i in range(1, len(gs_a1c_tfidf_Q9)+1)]
ndcg_a1c_tfidf_Q10 = [ndcg_at_k(gs_a1c_tfidf_Q10, i) for i in range(1, len(gs_a1c_tfidf_Q10)+1)]

# zip nDCG results of the 3 models
out_cre_Q1 = zip(gs_cre_allModels_Q1, ndcg_cre_allModels_Q1, ndcg_cre_elmo_Q1, ndcg_cre_tfidf_Q1)
out_cre_Q2 = zip(gs_cre_allModels_Q2, ndcg_cre_allModels_Q2, ndcg_cre_elmo_Q2, ndcg_cre_tfidf_Q2)
out_cre_Q3 = zip(gs_cre_allModels_Q3, ndcg_cre_allModels_Q3, ndcg_cre_elmo_Q3, ndcg_cre_tfidf_Q3)
out_cre_Q4 = zip(gs_cre_allModels_Q4, ndcg_cre_allModels_Q4, ndcg_cre_elmo_Q4, ndcg_cre_tfidf_Q4)
out_cre_Q5 = zip(gs_cre_allModels_Q5, ndcg_cre_allModels_Q5, ndcg_cre_elmo_Q5, ndcg_cre_tfidf_Q5)
out_cre_Q6 = zip(gs_cre_allModels_Q6, ndcg_cre_allModels_Q6, ndcg_cre_elmo_Q6, ndcg_cre_tfidf_Q6)
out_cre_Q7 = zip(gs_cre_allModels_Q7, ndcg_cre_allModels_Q7, ndcg_cre_elmo_Q7, ndcg_cre_tfidf_Q7)
out_cre_Q8 = zip(gs_cre_allModels_Q8, ndcg_cre_allModels_Q8, ndcg_cre_elmo_Q8, ndcg_cre_tfidf_Q8)
out_cre_Q9 = zip(gs_cre_allModels_Q9, ndcg_cre_allModels_Q9, ndcg_cre_elmo_Q9, ndcg_cre_tfidf_Q9)
out_cre_Q10 = zip(gs_cre_allModels_Q10, ndcg_cre_allModels_Q10, ndcg_cre_elmo_Q10, ndcg_cre_tfidf_Q10)

out_glu_Q1 = zip(gs_glu_allModels_Q1, ndcg_glu_allModels_Q1, ndcg_glu_elmo_Q1, ndcg_glu_tfidf_Q1)
out_glu_Q2 = zip(gs_glu_allModels_Q2, ndcg_glu_allModels_Q2, ndcg_glu_elmo_Q2, ndcg_glu_tfidf_Q2)
out_glu_Q3 = zip(gs_glu_allModels_Q3, ndcg_glu_allModels_Q3, ndcg_glu_elmo_Q3, ndcg_glu_tfidf_Q3)
out_glu_Q4 = zip(gs_glu_allModels_Q4, ndcg_glu_allModels_Q4, ndcg_glu_elmo_Q4, ndcg_glu_tfidf_Q4)
out_glu_Q5 = zip(gs_glu_allModels_Q5, ndcg_glu_allModels_Q5, ndcg_glu_elmo_Q5, ndcg_glu_tfidf_Q5)
out_glu_Q6 = zip(gs_glu_allModels_Q6, ndcg_glu_allModels_Q6, ndcg_glu_elmo_Q6, ndcg_glu_tfidf_Q6)
out_glu_Q7 = zip(gs_glu_allModels_Q7, ndcg_glu_allModels_Q7, ndcg_glu_elmo_Q7, ndcg_glu_tfidf_Q7)
out_glu_Q8 = zip(gs_glu_allModels_Q8, ndcg_glu_allModels_Q8, ndcg_glu_elmo_Q8, ndcg_glu_tfidf_Q8)
out_glu_Q9 = zip(gs_glu_allModels_Q9, ndcg_glu_allModels_Q9, ndcg_glu_elmo_Q9, ndcg_glu_tfidf_Q9)
out_glu_Q10 = zip(gs_glu_allModels_Q10, ndcg_glu_allModels_Q10, ndcg_glu_elmo_Q10, ndcg_glu_tfidf_Q10)

out_a1c_Q1 = zip(gs_a1c_allModels_Q1, ndcg_a1c_allModels_Q1, ndcg_a1c_elmo_Q1, ndcg_a1c_tfidf_Q1)
out_a1c_Q2 = zip(gs_a1c_allModels_Q2, ndcg_a1c_allModels_Q2, ndcg_a1c_elmo_Q2, ndcg_a1c_tfidf_Q2)
out_a1c_Q3 = zip(gs_a1c_allModels_Q3, ndcg_a1c_allModels_Q3, ndcg_a1c_elmo_Q3, ndcg_a1c_tfidf_Q3)
out_a1c_Q4 = zip(gs_a1c_allModels_Q4, ndcg_a1c_allModels_Q4, ndcg_a1c_elmo_Q4, ndcg_a1c_tfidf_Q4)
out_a1c_Q5 = zip(gs_a1c_allModels_Q5, ndcg_a1c_allModels_Q5, ndcg_a1c_elmo_Q5, ndcg_a1c_tfidf_Q5)
out_a1c_Q6 = zip(gs_a1c_allModels_Q6, ndcg_a1c_allModels_Q6, ndcg_a1c_elmo_Q6, ndcg_a1c_tfidf_Q6)
out_a1c_Q7 = zip(gs_a1c_allModels_Q7, ndcg_a1c_allModels_Q7, ndcg_a1c_elmo_Q7, ndcg_a1c_tfidf_Q7)
out_a1c_Q8 = zip(gs_a1c_allModels_Q8, ndcg_a1c_allModels_Q8, ndcg_a1c_elmo_Q8, ndcg_a1c_tfidf_Q8)
out_a1c_Q9 = zip(gs_a1c_allModels_Q9, ndcg_a1c_allModels_Q9, ndcg_a1c_elmo_Q9, ndcg_a1c_tfidf_Q9)
out_a1c_Q10 = zip(gs_a1c_allModels_Q10, ndcg_a1c_allModels_Q10, ndcg_a1c_elmo_Q10, ndcg_a1c_tfidf_Q10)

# store nDCG results as Dataframe
df_cre_Q1 = pd.DataFrame(out_cre_Q1, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q2 = pd.DataFrame(out_cre_Q2, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q3 = pd.DataFrame(out_cre_Q3, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q4 = pd.DataFrame(out_cre_Q4, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q5 = pd.DataFrame(out_cre_Q5, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q6 = pd.DataFrame(out_cre_Q6, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q7 = pd.DataFrame(out_cre_Q7, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q8 = pd.DataFrame(out_cre_Q8, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q9 = pd.DataFrame(out_cre_Q9, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_cre_Q10 = pd.DataFrame(out_cre_Q10, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])

df_glu_Q1 = pd.DataFrame(out_glu_Q1, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q2 = pd.DataFrame(out_glu_Q2, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q3 = pd.DataFrame(out_glu_Q3, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q4 = pd.DataFrame(out_glu_Q4, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q5 = pd.DataFrame(out_glu_Q5, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q6 = pd.DataFrame(out_glu_Q6, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q7 = pd.DataFrame(out_glu_Q7, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q8 = pd.DataFrame(out_glu_Q8, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q9 = pd.DataFrame(out_glu_Q9, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_glu_Q10 = pd.DataFrame(out_glu_Q10, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])

df_a1c_Q1 = pd.DataFrame(out_a1c_Q1, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q2 = pd.DataFrame(out_a1c_Q2, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q3 = pd.DataFrame(out_a1c_Q3, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q4 = pd.DataFrame(out_a1c_Q4, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q5 = pd.DataFrame(out_a1c_Q5, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q6 = pd.DataFrame(out_a1c_Q6, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q7 = pd.DataFrame(out_a1c_Q7, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q8 = pd.DataFrame(out_a1c_Q8, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q9 = pd.DataFrame(out_a1c_Q9, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])
df_a1c_Q10 = pd.DataFrame(out_a1c_Q10, columns=['Gold Standard', 'All-Models', 'ELMo', 'TF-IDF'])

#cre_mean = pd.Panel({n: df for n, df in enumerate([df_Q1, df_Q2, df_Q3, df_Q4, df_Q5, df_Q6, df_Q7, df_Q8, df_Q9, df_Q10])})
#print(cre_mean.mean(axis=0))
#df_a1c_Q10.to_csv('/Users/luyu/Documents/Master Thesis/rating/test.csv', index=False)
#glu_mean = pd.Panel({n: df for n, df in enumerate([df_glu_Q1, df_glu_Q2, df_glu_Q3, df_glu_Q4, df_glu_Q5, df_glu_Q6,
#                                                   df_glu_Q7, df_glu_Q8, df_glu_Q9, df_glu_Q10])})
#glu_mean.mean(axis=0).to_csv('/Users/luyu/Documents/Master Thesis/rating/test.csv', index=False)

#a1c_mean = pd.Panel({n: df for n, df in enumerate([df_a1c_Q1, df_a1c_Q2, df_a1c_Q3, df_a1c_Q4, df_a1c_Q5, df_a1c_Q6,
#                                                   df_a1c_Q7, df_a1c_Q8, df_a1c_Q9, df_a1c_Q10])})
#a1c_mean.mean(axis=0).to_csv('/Users/luyu/Documents/Master Thesis/rating/test.csv', index=False)