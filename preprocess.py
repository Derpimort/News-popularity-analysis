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
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import dateparser
from attribute_lists import TITLE_L, KEYWORD_L, DESC_L, AUTHOR_L, PUBLISHED_L, CONTENT_L

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
    
class Article(ArticleText):
    def __init__(self, url, raw=None):
        if not raw:
            raw=requests.get(url).content
        soup=BeautifulSoup(raw, 'lxml')
        self.url=url
        self.metadata=self.getMeta(soup)
        content=self.metadata['content'].find("section")
        if not content:
            content=self.metadata['content']
        super().__init__(self.metadata['title'], " ".join(list(content.stripped_strings)))
        
    def iterTillHit(self, soup, arglist, target=None):
        for arg in arglist:
            cont=soup.find(*arg)
            if cont:
                if not target:
                    return cont
                elif cont.text:
                    return cont.text
                else:
                    return cont[target]
        else:
            return None
        
    def getMeta(self, soup):
        # Title, Keywords, Description, Author, Published
        attr_d={}
        attr_d['title']=self.iterTillHit(soup, TITLE_L, 'content')
        attr_d['keyword']=self.iterTillHit(soup, KEYWORD_L, 'content')
        attr_d['desc']=self.iterTillHit(soup, DESC_L, 'content')
        attr_d['author']=self.iterTillHit(soup, AUTHOR_L, 'content')
        attr_d['published']=self.iterTillHit(soup, PUBLISHED_L, 'content')
        attr_d['content']=self.iterTillHit(soup, CONTENT_L)
        
        return attr_d
    
    def num_hrefs(self):
        return len(self.metadata['content'].findAll("a", href=True))
    
    def num_self_hrefs(self):
        site=urlparse(self.url)[1]
        return sum([1 for href in self.metadata['content'].findAll("a", href=True) if site in href['href']])
    
    def num_imgs(self):
        return len(self.metadata['content'].findAll("img"))
    
    def num_videos(self):
        return len(self.metadata['content'].findAll("iframe"))
    
    def num_keywords(self):
        return len(self.metadata['keyword'].split(",")) if self.metadata['keyword'] else 0
    
    def daystuff(self):
        weekday_dict=[["weekday_is_monday",0],
            ["weekday_is_tuesday",0],
            ["weekday_is_wednesday",0],
            ["weekday_is_thursday",0],
            ["weekday_is_friday",0],
            ["weekday_is_saturday",0],
            ["weekday_is_sunday",0],
            ["is_weekend",0]]
        
        try:
            weekday=dateparser.parse(self.metadata['published']).weekday()
            weekday_dict[weekday][1]=1
            weekday_dict[-1][1]=1 if weekday>4 else 0
        except TypeError:
            pass
        finally:
            return dict(weekday_dict)
        
    def stats(self):
        attributes=['num_hrefs', 'num_self_hrefs',
                    'num_imgs', 'num_videos', 'num_keywords']
        meta_dict=super().stats()
        meta_dict.update({func:getattr(self, func)() for func in attributes})
        meta_dict.update(self.daystuff())
        return meta_dict