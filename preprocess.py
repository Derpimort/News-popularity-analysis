#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:57:11 2020

@author: darp_lord
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from textblob import TextBlob

class ArticleText:
    vader=SentimentIntensityAnalyzer()
    stopwords_l=set(stopwords.words('english'))
    
    def __init__(self, title, content):
        sentiment_l=[]
        self.non_stop_w=[]
        self.wordlen=0
        self.title_l=re.findall(r'\w+', title)
        
        content_l=re.findall(r'\w+', content)
        self.content_tokens=len(content_l)
        self.unique_tokens=len(np.unique(content_l))
        for word in content_l:
            sentiment_l.append(ArticleText.vader.polarity_scores(word))
            sentiment_l[-1].update(dict(TextBlob(word).sentiment._asdict()))
            self.wordlen+=len(word)
            if word not in ArticleText.stopwords_l:
                self.non_stop_w.append(word)
        self.sentiment_df=pd.DataFrame(sentiment_l)
        self.title_sentiment=TextBlob(title).sentiment
        self.content_sentiment=TextBlob(content).sentiment
    
    def n_tokens_title(self):
        return len(self.title_l)
    
    def n_tokens_content(self):
        return self.content_tokens
    
    def n_unique_tokens(self):
        return self.unique_tokens/self.content_tokens
    
    def n_non_stop_words(self):
        return len(self.non_stop_w)/self.content_tokens
    
    def n_non_stop_unique_tokens(self):
        return len(np.unique(self.non_stop_w))/len(self.non_stop_w)
    
    def average_token_length(self):
        return self.wordlen/self.content_tokens
    
    def global_subjectivity(self):
        return self.content_sentiment.subjectivity
    
    def global_sentiment_polarity(self):
        return self.content_sentiment.polarity
    
    def global_rate_positive_words(self):
        return self.sentiment_df['pos'].sum()/self.content_tokens
    
    def global_rate_negative_words(self):
        return self.sentiment_df['neg'].sum()/self.content_tokens
    
    def rate_positive_words(self):
        return self.sentiment_df['pos'].sum()/(self.sentiment_df['neu']!=1).sum()
    
    def rate_negative_words(self):
        return self.sentiment_df['neg'].sum()/(self.sentiment_df['neu']!=1).sum()
    
    def avg_positive_polarity(self):
        try:
            return self.sentiment_df[['pos', 'polarity']].groupby('pos').mean().loc[1].item()
        except KeyError:
            print("No positive words")
            return 0
    
    def min_positive_polarity(self):
        try:
            return self.sentiment_df[['pos', 'polarity']].groupby('pos').min().loc[1].item()
        except KeyError:
            print("No positive words")
            return 0
    
    def max_positive_polarity(self):
        try:
            return self.sentiment_df[['pos', 'polarity']].groupby('pos').max().loc[1].item()
        except KeyError:
            print("No positive words")
            return 0
    
    def avg_negative_polarity(self):
        try:
            return self.sentiment_df[['neg', 'polarity']].groupby('neg').mean().loc[1].item()
        except KeyError:
            print("No negative words")
            return 0
    
    def min_negative_polarity(self):
        try:
            return self.sentiment_df[['neg', 'polarity']].groupby('neg').min().loc[1].item()
        except KeyError:
            print("No negative words")
            return 0
    
    def max_negative_polarity(self):
        try:
            return self.sentiment_df[['neg', 'polarity']].groupby('neg').max().loc[1].item()
        except KeyError:
            print("No negative words")
            return 0
    
    def title_subjectivity(self):
        return self.title_sentiment.subjectivity
    
    def title_sentiment_polarity(self):
        return self.title_sentiment.polarity

    def stats(self):
        attributes=['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 
       'n_non_stop_words', 'n_non_stop_unique_tokens',
       'average_token_length', 
       'global_subjectivity', 'global_sentiment_polarity', 
       'global_rate_positive_words', 'global_rate_negative_words', 
       'rate_positive_words','rate_negative_words', 
       'avg_positive_polarity','min_positive_polarity', 'max_positive_polarity',
       'avg_negative_polarity', 'min_negative_polarity', 'max_negative_polarity', 
       'title_subjectivity','title_sentiment_polarity']
        
        return {func:getattr(self, func)() for func in attributes}