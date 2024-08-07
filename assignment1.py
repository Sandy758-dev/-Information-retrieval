import os
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import text as stoptext
import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib.style

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 18})

document_file = 'cran.all'
query_file = 'query.text'
qrels_file = 'qrels.text'

class CustomIDFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        #  doc freq
        self.document_frequencies_ = np.sum(X != 0, axis=0)
        return self
    
    def transform(self, X):
        # term freq
        term_frequencies = X
        
        # Calculate inverse doc freq
        inverse_document_frequencies = np.where(self.document_frequencies_ != 0, 1 / self.document_frequencies_, 1)
        
       
        inverse_document_frequencies = np.squeeze(inverse_document_frequencies)
        
        # custom TF-IDF
        tfidf = term_frequencies.multiply(inverse_document_frequencies)
        
        return tfidf

###########################Helper methods#######################
def get_prf(relevant_docs, retrieved_docs):
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for doc in relevant_docs:
        if doc-1 in retrieved_docs:
            true_positive += 1
        else:
            false_negative += 1 

    for doc in retrieved_docs:
        if doc+1 not in relevant_docs:
            false_positive += 1

    p = true_positive / (true_positive + false_positive)
    r = true_positive / (true_positive + false_negative)
    
    if p == 0 or r == 0:
        f = 0
    else:
        f = 2 * ((p * r)/(p + r))
    
    return p, r, f

def get_model_result(top10_cosine, top10_manhattan, cosine_prf, manhattan_prf):
    
    for query in range(0, len(top10_cosine)):
        relevant_docs = qrels_dict[query+1]
        retrieved_docs_cosine = top10_cosine[query]
        retrieved_docs_manhattan = top10_manhattan[query]

        p,r,f = get_prf(relevant_docs, retrieved_docs_cosine)
        cosine_prf['p'].append(p)
        cosine_prf['r'].append(r)
        cosine_prf['f'].append(f)

        p,r,f = get_prf(relevant_docs, retrieved_docs_manhattan)
        manhattan_prf['p'].append(p)
        manhattan_prf['r'].append(r)
        manhattan_prf['f'].append(f)


def get_console_output(cosine_prf, manhattan_prf):

    result = {'f':{}, 'p':{}, 'r':{}}
    for key in result:
        result[key]['cos'] = (np.array(cosine_prf[key]).mean(), np.array(cosine_prf[key]).max())
        result[key]['man'] = (np.array(manhattan_prf[key]).mean(), np.array(manhattan_prf[key]).max())

    return result

###########################################################################



#####################Parse each document#######################
document_list = []
doc_ID = re.compile(r'\.I\s+')
with open (document_file, 'r') as f:
    text = f.read().replace('\n'," ")
    document_list = re.split(doc_ID, text)
    document_list.pop(0) #first entry is empty so removed

body_start = re.compile(r'\.W\s+')
document_body_list = []
 
for line in document_list:
    # print(line)
    entries= re.split(body_start, line)
    body = entries[1].strip().lower()
    document_body_list.append(body)
##############################################################

######################Parse each query##########################
query_raw_list = []
query_ID = re.compile(r'\.I\s+')
with open (query_file, 'r') as f:
    text = f.read().replace('\n'," ")
    query_raw_list = re.split(query_ID, text)
    query_raw_list.pop(0) #first entry is empty so removed

query_list = []
for line in query_raw_list:
    # print(line)
    entries= re.split(body_start, line)
    body = entries[1].strip().lower()
    query_list.append(body)
#############################################################

stop_words = list(stoptext.ENGLISH_STOP_WORDS)

# Binary Vectorizer
binary_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(binary=True, stop_words=stop_words, lowercase=True)),
])

# Custom TF-IDF Vectorizer
custom_tfidf_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=stop_words, lowercase=True)),
    ('tfidf', CustomIDFTransformer())
])


# Fit lowercase data 

binary_vectorized_corpus = binary_pipeline.fit_transform(document_body_list)
custom_tfidf_vectorized_corpus = custom_tfidf_pipeline.fit_transform(document_body_list)
binary_vectorized_queries = binary_pipeline.transform(query_list)
custom_tfidf_vectorized_queries = custom_tfidf_pipeline.transform(query_list)

# Cal similarities
cosine_similarities_binary = cosine_similarity(binary_vectorized_queries, binary_vectorized_corpus)
manhattan_distances_binary = manhattan_distances(binary_vectorized_queries, binary_vectorized_corpus)
cosine_similarities_tfidf = cosine_similarity(custom_tfidf_vectorized_queries, custom_tfidf_vectorized_corpus)
manhattan_distances_tfidf = manhattan_distances(custom_tfidf_vectorized_queries, custom_tfidf_vectorized_corpus)

# Top 10 relevant documents 
top_10_binary_cosine = []
top_10_binary_manhattan = []
top_10_tfidf_cosine = []
top_10_tfidf_manhattan = []

for i in range(len(query_list)):
    
    sorted_binary_cosine = np.argsort(cosine_similarities_binary[i])[::-1][:10]
    sorted_binary_manhattan = np.argsort(manhattan_distances_binary[i])[:10]
    sorted_tfidf_cosine = np.argsort(cosine_similarities_tfidf[i])[::-1][:10]
    sorted_tfidf_manhattan = np.argsort(manhattan_distances_tfidf[i])[:10]
    
   
    top_10_binary_cosine.append(sorted_binary_cosine.tolist())
    top_10_binary_manhattan.append(sorted_binary_manhattan.tolist())
    top_10_tfidf_cosine.append(sorted_tfidf_cosine.tolist())
    top_10_tfidf_manhattan.append(sorted_tfidf_manhattan.tolist())

#####Prepare qrels(query relevant documents)############
qrels_text = open(qrels_file, 'r')
qrels_lines = qrels_text.readlines()
qrels_dict = {}

qrels_separator = re.compile(r'\s+')

for line in qrels_lines:
    data = [int(val) for val in re.split(qrels_separator, line.strip())]
    if data[0] in qrels_dict:
        qrels_dict[data[0]].append(data[1])
    else:
        qrels_dict[data[0]] = [data[1]]

# print(qrels_dict)
##############################

##################Console output########################

binary_cosine_prf = {'f':[], 'p':[], 'r':[]}
binary_manhattan_prf = {'f':[], 'p':[], 'r':[]}
custom_tfidf_cosine_prf = {'f':[], 'p':[], 'r':[]}
custom_tfidf_manhattan_prf = {'f':[], 'p':[], 'r':[]}

get_model_result(top_10_binary_cosine, top_10_binary_manhattan, binary_cosine_prf, binary_manhattan_prf)
get_model_result(top_10_tfidf_cosine, top_10_tfidf_manhattan, custom_tfidf_cosine_prf, custom_tfidf_manhattan_prf)

console_output = {}
console_output['Binary'] = get_console_output(binary_cosine_prf, binary_manhattan_prf)
console_output['TFIDF'] = get_console_output(custom_tfidf_cosine_prf, custom_tfidf_manhattan_prf)

pprint.pprint(console_output)

########################################################

####################Graphs##############################

for metric in ['Precision', 'Recall', 'F-Score']:

    r = np.arange(225)
    
    #Binary cosine
    plt.clf()
    plt.bar(r, binary_cosine_prf[metric[0].lower()], color='b', label='Binary (Cosine)')
    plt.legend()
    plt.title('{} of each query for Binary (using cosine similarity)'.format(metric))
    plt.xlabel('Query index')
    plt.ylabel(metric)
    plt.savefig('{}_Binary_cosine.png'.format(metric[0].lower()))

    #Binary manhattan
    plt.clf()
    plt.bar(r, binary_manhattan_prf[metric[0].lower()], color='b', label='Binary (Manhattan)')
    plt.legend()
    plt.title('{} of each query for Binary (using manhattan similarity)'.format(metric))
    plt.xlabel('Query index')
    plt.ylabel(metric)
    plt.savefig('{}_Binary_manhattan.png'.format(metric[0].lower()))

    #TF-IDF cosine
    plt.clf()
    plt.bar(r, custom_tfidf_cosine_prf[metric[0].lower()], color='b', label='TF-IDF (Cosine)')
    plt.legend()
    plt.title('{} of each query for TF-IDF (using cosine similarity)'.format(metric))
    plt.xlabel('Query index')
    plt.ylabel(metric)
    plt.savefig('{}_TF-IDF_cosine.png'.format(metric[0].lower()))

    #TF-IDF manhattan
    plt.clf()
    plt.bar(r, custom_tfidf_manhattan_prf[metric[0].lower()], color='b', label='TF-IDF (Manhattan)')
    plt.legend()
    plt.title('{} of each query for TF-IDF (using manhattan similarity)'.format(metric))
    plt.xlabel('Query index')
    plt.ylabel(metric)
    plt.savefig('{}_TF-IDF_manhattan.png'.format(metric[0].lower()))

##########################################################