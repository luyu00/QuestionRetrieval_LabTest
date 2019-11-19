import tensorflow as tf
import tensorflow_hub as hub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os, re, csv, nltk, logging, string


# read data from a csv file
def read_csv(filename, logout=True):
    try:
        reader = csv.reader(open(filename, "r"))
        data = []
        for r in reader:
            data.append(r)
        return data
    except Exception as e:
        if logout is True:
            logging.error(e)
        return None


# write data in format of [(x1,y1,z1),(x2,y2,z2)] to a csv file
def write_csv(filename, data, logout=True):
    try:
        doc = csv.writer(open(filename, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        for d in data:
            doc.writerow(d)
        return True
    except Exception as e:
        if logout is True:
            log.error(e)
        return False


# remove punctuation
def trans(s):
    exclude = string.punctuation
    return s.translate(str.maketrans({key: None for key in exclude}))


# chunk list to approximatly equal parts
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# calculate cosine similarity in 10*10 2D array
def tfidf_cosine_sim(model):
    l = []
    sim = []
    for x in range(model.shape[0]):
        l.append(x)
    for m in (l[:10]):
        for n in (l[10:]):
            sim.append(cosine_similarity(model[m], model[n])[0][0])
    return sim


# rank cosine similarity
def rank_question_similarities(model, corpus, corpus_i):
    cos = []
    for x in range(model.shape[0]):
        sim = cosine_similarity(model[0], model[x])
        cos.append([corpus_i[x], corpus[x], sim[0][0]])
    return cos


# calculate cosine similarity under Universal Sentence Encoder
def get_UniSentEnc_sim(corpus, corpus_i):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
    embed = hub.Module(module_url)
    init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    with tf.Session() as sess:
        sim = []
        sess.run([init, table_init])
        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        encoding_tensor = embed(similarity_input_placeholder)
        for f in range(len(corpus)):
            en_embeddings_1 = sess.run(encoding_tensor, feed_dict={similarity_input_placeholder: [corpus[0]]})
            en_embeddings_2 = sess.run(encoding_tensor, feed_dict={similarity_input_placeholder: [corpus[f]]})
            sim.append([corpus_i[f], corpus[f], float(cosine_similarity(en_embeddings_1, en_embeddings_2))])
    return sim


# calculate cosine similarity of ELmo
def get_elmo_sim(corpus, corpus_i):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    with tf.Session() as sess:
        sim = []
        sess.run([init, table_init])
        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        encoding_tensor = elmo(similarity_input_placeholder)
        for f in range(len(corpus)):
            en_embeddings_1 = sess.run(encoding_tensor, feed_dict={similarity_input_placeholder: [corpus[0]]})
            en_embeddings_2 = sess.run(encoding_tensor, feed_dict={similarity_input_placeholder: [corpus[f]]})
            sim.append([corpus_i[f], corpus[f], float(cosine_similarity(en_embeddings_1, en_embeddings_2))])
    return sim


# load mixed data
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q1.csv')
hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q2.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q3.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q4.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q5.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q6.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q7.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q8.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q9.csv')
#hba1c_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/HbA1c_top10mix_Q10.csv')

#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q1.csv')
noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q2.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q3.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q4.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q5.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q6.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q7.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q8.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q9.csv')
#noLab_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/NoLab_top10mix_Q10.csv')

glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q1.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q2.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q3.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q4.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q5.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q6.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q7.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q8.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q9.csv')
#glucose_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Glucose_top10mix_Q10.csv')

creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q1.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q2.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q3.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q4.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q5.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q6.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q7.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q8.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q9.csv')
#creatinine_mix_d = read_csv('/Users/luyu/Documents/Master Thesis/rating/mix_csv/Creatinine_top10mix_Q10.csv')

# extract id
hba1c_mix_id = [d[0] for d in hba1c_mix_d]
noLab_mix_id = [d[0] for d in noLab_mix_d]
glucose_mix_id = [d[0] for d in glucose_mix_d]
creatinine_mix_id = [d[0] for d in creatinine_mix_d]

# extract question and convert to lowercase
hba1c_mix = [d[1].lower() for d in hba1c_mix_d]
noLab_mix = [d[1].lower() for d in noLab_mix_d]
glucose_mix = [d[1].lower() for d in glucose_mix_d]
creatinine_mix = [d[1].lower() for d in creatinine_mix_d]

# remove punctuation
hba1c_mix_np = [trans(c) for c in hba1c_mix]
noLab_mix_np = [trans(c) for c in noLab_mix]
glucose_mix_np = [trans(c) for c in glucose_mix]
creatinine_mix_np = [trans(c) for c in creatinine_mix]

# tokenization
hba1c_mix_token = [word_tokenize(c) for c in hba1c_mix_np]
noLab_mix_token = [word_tokenize(c) for c in noLab_mix_np]
glucose_mix_token = [word_tokenize(c) for c in glucose_mix_np]
creatinine_mix_token = [word_tokenize(c) for c in creatinine_mix_np]

# remove stopwords
words_stop = [str(c) for c in stopwords.words('english')]
hba1c_mix_no_stopw = [[cc for cc in c if cc not in words_stop]for c in hba1c_mix_token]
noLab_mix_no_stopw = [[cc for cc in c if cc not in words_stop]for c in noLab_mix_token]
glucose_mix_no_stopw = [[cc for cc in c if cc not in words_stop]for c in glucose_mix_token]
creatinine_mix_no_stopw = [[cc for cc in c if cc not in words_stop]for c in creatinine_mix_token]

# stemming words
ps = nltk.stem.PorterStemmer()
hba1c_mix_stemw = [[str(ps.stem(s)) for s in sw] for sw in hba1c_mix_no_stopw]
noLab_mix_stemw = [[str(ps.stem(s)) for s in sw] for sw in noLab_mix_no_stopw]
glucose_mix_stemw = [[str(ps.stem(s)) for s in sw] for sw in glucose_mix_no_stopw]
creatinine_mix_stemw = [[str(ps.stem(s)) for s in sw] for sw in creatinine_mix_no_stopw]

# convert tokens back to sentence
hba1c_mix_stemw_s = [' '.join(i) for i in hba1c_mix_stemw]
noLab_mix_stemw_s = [' '.join(i) for i in noLab_mix_stemw]
glucose_mix_stemw_s = [' '.join(i) for i in glucose_mix_stemw]
creatinine_mix_stemw_s = [' '.join(i) for i in creatinine_mix_stemw]

# vectorize using tfidf
vectorizer = TfidfVectorizer(min_df=1)
Hm = vectorizer.fit_transform(hba1c_mix_stemw_s)
Nm = vectorizer.fit_transform(noLab_mix_stemw_s)
Gm = vectorizer.fit_transform(glucose_mix_stemw_s)
Cm = vectorizer.fit_transform(creatinine_mix_stemw_s)

#out = rank_question_similarities(Hm, hba1c_mix, hba1c_mix_id)
#out = rank_question_similarities(Nm,noLab_mix, noLab_mix_id)
#out = rank_question_similarities(Gm,glucose_mix, glucose_mix_id)
out = rank_question_similarities(Cm,creatinine_mix, creatinine_mix_id)
write_csv('/Users/luyu/Documents/Master Thesis/test.csv', out)