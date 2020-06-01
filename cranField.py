import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import nltk
import operator
import sklearn.metrics
from statistics import mean


keyFile = open('cran.qry', 'r')
key = keyFile.readlines()
key_dict = {}
responseFile = open('cranqrel', 'r')
response = responseFile.readlines()
response_dict = {}

keyFile = open('cran.qry', 'r')
key = keyFile.readlines()
for ii in key:
    print(ii+'\n')
# 365 queries

responseFile = open('cranqrel', 'r')
response = responseFile.readlines()
for i in response:
    print(i+'\n')
# query_number(1-225)  document_number  relevance_rank(-1-5)

keyF = open('cran.all.1400', 'r')
key2 = keyF.readlines()
for iii in key2:
    print(iii+'\n')

with open('cran.all.1400') as f:
    articles = f.read().split('\n.I')


def process(article):
    article = article.split('\n.T\n')[1]
    T, _, article = article.partition('\n.A\n')
    A, _, article = article.partition('\n.B\n')
    B, _, W = article.partition('\n.W\n')
    return {'T':T, 'A':A, 'B':B, 'W':W}


data = {(i+1):process(article) for i,article in enumerate(articles)}
docs = [data[index]['W'] for index in range(1, 1401)]# is a list of docs/abstracts that are strings.

nltk.download('stopwords')
Stop_Words = stopwords.words("english")


def tokenize(text):
    clean_txt = re.sub('[^a-z\s]+',' ',text)  # replacing spcl chars, punctuations by space
    clean_txt = re.sub('(\s+)',' ',clean_txt)  # replacing multiple spaces by single space
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(clean_txt))  # tokenizing, lowercase
    words = [word for word in words if word not in Stop_Words]  # filtering stopwords
    words = filter(lambda t: len(t)>=min_length, words)  # filtering words of length <=2
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words)))  # stemming tokens
    return tokens


vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=3,max_df=0.90,
                             use_idf=True, sublinear_tf=True,norm='l2')

termdocmtx = vectorizer.fit_transform(docs).toarray()
features_ = vectorizer.get_feature_names()
datfram = pd.DataFrame(features_)
term_doc = pd.DataFrame(termdocmtx)
term_doc.columns = features_

query = input("Enter the query\n")
qvec = vectorizer.transform([query]).toarray()

termdocmtx = vectorizer.fit_transform(docs).toarray()
features_ = vectorizer.get_feature_names()
sim_scores = sklearn.metrics.pairwise.cosine_similarity(termdocmtx,qvec)
doc_sim={}
k = input("enter the number of results expected\n")
k = int(k)
for item in range(1400):
  doc_sim[item] = sim_scores[item]


sort_dict = sorted(doc_sim.items(), key=operator.itemgetter(1), reverse=True)
results_ = [sort_dict[i][0] for i in range(k)]
results_topk = [docs[u] for u in results_]
print(results_topk[0])


with open('cran.qry') as fq:
    queries = fq.read().split('\n.I')


def process(query):
    query = query.split('\n.W\n')[1]
    W, _, s = query.partition('\n.W\n')
    return W


dataq = {(ind+1):process(query) for ind,query in enumerate(queries)}
l_l = [response[c].split() for c in range(len(response))]
lol = [p[:2] for p in l_l]  #lol = listOf [query no. , relevant doc_no.]
kkk = [[]]*225


kz = [0]*225
for cc in lol:
    ccc = int(cc[0])-1
    kz[ccc] = kz[ccc]+1
# elements of kz = |rel docs| for that query number = index of kz


def sum_el(arr,nn):
    ss = 0
    for ell in range(nn):
        ss = ss+arr[ell]
    return ss


for uz in range(225):
    tmp = [lol[uu][1] for uu in range(sum_el(kz,uz),sum_el(kz,uz) + kz[uz])]
    kkk[0] = tmp
precision = [0]*225


def common(a,b):
    c = [value for value in a if value in b]
    return len(c)


for qq in range(225):
    query = dataq[qq+1]
    qvec = vectorizer.transform([query]).toarray()
    sim_scores = sklearn.metrics.pairwise.cosine_similarity(termdocmtx,qvec)
    doc_sim={}
    k = kz[qq]
    for item in range(1400):
        doc_sim[item] = sim_scores[item]
    sort_dict = sorted(doc_sim.items(), key = operator.itemgetter(1), reverse=True)
    results_ = [sort_dict[i][0] for i in range(k)]
    ttmp = [str(tmpp) for tmpp in results_]
    r = common(kkk[qq],ttmp)
    precision[qq] = r/k


mean(precision)
max(precision)
min(precision)
