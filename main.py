#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:08:23 2020

@author: darp_lord
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle
import preprocess
from tqdm import tqdm

MODEL_DIR="./models/"

def getPreds(data):
    with open(MODEL_DIR+"xgb.model", 'rb') as f:
        clf=pickle.load(f)
    return clf.predict(data)

def predictVirality(urls):
    data=[]
    loop=tqdm(urls)
    for url in loop:
        try:
            ret_d={'url':url}
            ret_d.update(preprocess.Article(url).stats())
            data.append(ret_d)
        except Exception as e:
            print("Could not process: ",url)
            print(e)
    data=pd.DataFrame(data)
    #print(data)
    for index, val in enumerate(getPreds(data.iloc[:,1:].values)):
        print(data['url'][index],val)
        
if __name__=="__main__":
    urls=[]
    url="empty"
    while(url!=""):
        url=input("Enter url (empty to end): ")
        urls.append(url)
    predictVirality(urls[0:-1])