from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex as InvColab
import numpy as np
import pandas as pd
from collections import Counter
import pickle
from pathlib import Path
import json
import math
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from google.cloud import storage

nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
bucket_name = 'assaignment316093632'
client = storage.Client()
blobs = client.list_blobs(bucket_name)
bucket = client.get_bucket('assaignment316093632')

blob = bucket.get_blob('body/index.pkl')
with blob.open("rb") as f:
    bodyInx = pickle.load(f)

blob = bucket.get_blob('title/title_index.pkl')
with blob.open("rb") as f:
    titleInx = pickle.load(f)

blob = bucket.get_blob('anchor/anchor_text_index.pkl')
with blob.open("rb") as f:
    anchorInx = pickle.load(f)

blob = bucket.get_blob('metadata/dl.pkl')
with blob.open("rb") as f:
    DL = pickle.load(f)

blob = bucket.get_blob('metadata/pageviews-202108-user.pkl')
with blob.open("rb") as f:
    viewsPKL = pickle.load(f)

blob = bucket.get_blob('metadata/titels.pkl')
with blob.open("rb") as f:
    titels = pickle.load(f)

blob = bucket.get_blob('metadata/pagerank.pkl')
with blob.open("rb") as f:
    pagerank = pickle.load(f)

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def merge_results(body_rank,title_rank,title_weight=0.83,text_weight=0.17):    
    mergeDict = {}
    for doc in body_rank:
        mergeDict[doc[0]] = text_weight * doc[1]
    for doc in title_rank:
        if doc[0] in mergeDict.keys():
            mergeDict[doc[0]] += title_weight * doc[1]
        else:
            mergeDict[doc[0]] = title_weight * doc[1]
    res = (doc_id for doc_id, score in sorted(mergeDict.items(),key=lambda x : x[1] , reverse=False))
    return res

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    base_dir = 'title'
    words, pls = zip(*titleInx.posting_lists_iter(base_dir, query.lower().split()))
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = []
    for tok in tokens:
        if tok not in english_stopwords or tok in bodyInx.df.keys():
          query.append(tok)
          
    docsBody = {}
    docsTitle = {}
    for word in query:
      if(word in words):
        for doc_id, score in sorted((tuple((doc_id, score)) for doc_id, score in pls[words.index(word)]), key = lambda x: x[1],reverse=True):
          if(doc_id not in docsBody.keys()):
            docsBody[doc_id] = 1
          else:
            docsBody[doc_id] += 1
          if(doc_id not in docsTitle.keys()):
            docsTitle[doc_id] = 1
          else:
            docsTitle[doc_id] += 1
    bodyList = []
    for doc in docsBody.keys():
      bodyList.append(tuple((doc, docsBody[doc])))
    titleList = []
    for doc in docsTitle.keys():
      titleList.append(tuple((doc, docsTitle[doc])))
      
    for doc in merge_results(bodyList,titleList):
      try:
        res.append((doc,titels[doc]))
      except:
        res.append((doc,'NoTitleFound'))

    # END SOLUTION
    return jsonify(res[:100])

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    base_dir = 'body'
    words, pls = zip(*bodyInx.posting_lists_iter(base_dir, query.split()))
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = []
    for tok in tokens:
        if tok not in english_stopwords or tok in bodyInx.df.keys():
          query.append(tok)
    D = generate_document_tfidf_matrix(query,bodyInx,words,pls)
    Q = generate_query_tfidf_vector(query,bodyInx)
    topN = get_top_n(cosine_similarity(D,Q),100)
    
    for doc in topN[:100]:
      try:
        res.append((doc[0],titels[doc[0]]))
      except:
        res.append((doc[0],'NoTitleFound'))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    base_dir = 'title'
    words, pls = zip(*titleInx.posting_lists_iter(base_dir, query.split()))
    docs = {}
    for word in query.split():
      if(word in words):
        for doc_id, score in sorted((tuple((doc_id, score)) for doc_id, score in pls[words.index(word)]), key = lambda x: x[1],reverse=True):
          if(doc_id not in docs.keys()):
            docs[doc_id] = 1
          else:
            docs[doc_id] += 1

            
    for doc in sorted(docs.items(), key=lambda x: x[1], reverse=True):
      try:
        res.append((doc[0],titels[doc[0]]))
      except:
        res.append((doc[0],'NoTitleFound'))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    base_dir = 'anchor'
    words, pls = zip(*anchorInx.posting_lists_iter(base_dir, query.split()))
    docs = {}
    for word in query.split():
      if(word in words):
        for doc_id, score in sorted((tuple((doc_id, score)) for doc_id, score in pls[words.index(word)]), key = lambda x: x[1],reverse=True):
          if(doc_id not in docs.keys()):
            docs[doc_id] = 1 #or score?
          else:
            docs[doc_id] += 1
    
    for doc in sorted(docs.items(), key=lambda x: x[1], reverse=True):
      try:
        res.append((doc[0],titels[doc[0]]))
      except:
        res.append((doc[0],'NoTitleFound'))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    wiki_ids = json.loads(wiki_ids['json'])
    
    for doc_id in wiki_ids:
      try:
        res.append(float(pagerank[str(doc_id)]))
      except:
        res.append(float(0))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    wiki_ids = json.loads(wiki_ids['json'])
    
    for doc_id in wiki_ids:
      try:
        res.append(viewsPKL[doc_id])
      except:
        res.append(0)
    return jsonify(res)

def generate_query_tfidf_vector(query_to_search, index):
    """
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.
    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())    
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys(): #avoid terms that do not appear in the index.               
            tf = counter[token]/len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]            
            idf = math.log((len(DL))/(df+epsilon),10) #smoothing
            
            try:
                ind = term_vector.index(token)
                Q[ind] = tf*idf                    
            except:
                pass
    return Q
    

def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.
    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(DL)        
    for term in np.unique(query_to_search):        
        if term in words:            
            list_of_doc = pls[words.index(term)]   
            normlized_tfidf = []     
            for doc_id, freq in list_of_doc:
              normlized_tfidf.append((doc_id,(freq/DL[doc_id])*math.log(N/index.df[term],10)))    
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.
    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """
    print("term total")
    print(len(index.term_total))
    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score
    Parameters:
    -----------
    D: DataFrame of tfidf scores.
    Q: vectorized query with tfidf scores
    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    simDict = {}
    for i in D.iterrows():
      simDict[i[0]] = np.dot(i[1], Q)/(np.linalg.norm(i[1])*np.linalg.norm(Q))
    return simDict
 
def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))
    N: Integer (how many documents to retrieve). By default N = 3
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
