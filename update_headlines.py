#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:43:59 2020

@author: darp_lord
"""

import os
import numpy as np
from newsapi import NewsApiClient
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, TfidfModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from gensim.test.utils import datapath

MODEL_DIR="./models/"
lemma = WordNetLemmatizer()

def checkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def lemma_pp(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemma.lemmatize(token))
    return result


def updateLDA():
    api_file="./newsapi.key"
    categories=['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
    
    with open(api_file,"r") as apikey:
        newsapi=NewsApiClient(api_key=apikey.read().strip())
    
    headlines={cat:newsapi.get_top_headlines(category=cat, language='en', country='in') for cat in categories}
    pp_docs=[]
    
    for category in headlines:
        for article in headlines[category]['articles']:
            #print(lemma_pp(article['title']))
            pp_docs.append(lemma_pp(article['title']))
            
            
    if os.path.exists(MODEL_DIR+"corpus_dict.model"):
        corp_d=Dictionary.load(MODEL_DIR+"corpus_dict.model")
        corp_d.add_documents(pp_docs)
    else:
        corp_d = Dictionary(pp_docs)
        corp_d.filter_extremes(no_below=2, no_above=0.5)
    
    
    dtm=[corp_d.doc2bow(doc) for doc in pp_docs]
    
    tfidf=TfidfModel(dtm)
    corp_tfidf=tfidf[dtm]
    
    lda = LdaMulticore(corp_tfidf, num_topics=14, id2word=corp_d, passes=60, workers=3)
    print(lda.print_topics(num_topics=14, num_words=5))
    checkdir(MODEL_DIR)
    corp_d.save(MODEL_DIR+"corpus_dict.model")
    #corp_tfidf.save(MODEL_DIR+"corpus_tfidf.model")
    lda.save(MODEL_DIR+"lda.model")
    
def getLDA(topics):
    corp_d=Dictionary.load(MODEL_DIR+"corpus_dict.model")
    
    lda=LdaMulticore.load(MODEL_DIR+"lda.model")
    pp_docs=[]
    for topic in topics:
        pp_docs.append(lemma_pp(topic))
    dtm=[corp_d.doc2bow(doc) for doc in pp_docs]
    tfidf=TfidfModel(dtm)
    corp_tfidf=tfidf[dtm]
    return list(lda[tfidf[dtm]])

if __name__=="__main__":
    #updateLDA()
    print({"LDA_%.2d"%i:j for i,j in getLDA(["Prime Minister Ardern says New Zealand has won \"battle\" against community spread of coronavirus - CBS News"])[0]})
