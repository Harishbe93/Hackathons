# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:00:25 2018

@author: S5RXCY
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
from wordcloud import WordCloud,STOPWORDS
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import time
from sklearn.model_selection import KFold, RepeatedKFold,train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

os.getcwd()
os.chdir("\\\\CHRB1067.CORP.GWPNET.COM\\homes\\C\\S5RXCY\\Documents\\\\HactionLab\\Task 3 - LCAT classification")


train = pd.read_excel("Task3 (2).xlsx")
os.chdir("DBpedia")
for i in range(len(train["Disaster Name"])):
    try:
       with open(train["Wikipedia Link "][i].replace("https://en.wikipedia.org/wiki/","")+".html","r",encoding = "utf-8") as f:
            data = f.read()
            soup = BeautifulSoup(data)
            if soup.find("span",{'property':'dbo:abstract','xml:lang':'en'}):
                train.loc[i,"Description"] = soup.find("span",{'property':'dbo:abstract','xml:lang':'en'}).text
    except OSError as e:
        pass
    finally:
        pass


labels = list(train["Lcat_Scenario"].str.strip().unique())
index = list(range(len(labels)))
zipped = zip(labels,index)
mapping = dict(zipped)

train["target"] = train["Lcat_Scenario"].str.strip().map(mapping)

print('Preprocessing text...')
cols = [
   'Description'
]
n_features = [
    400
]
train["Description"] = train["Description"].astype(str)
for c_i, c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i], min_df=3)
    tfidf.fit(train[c])
    tfidf_train = np.array(tfidf.transform(train[c]).todense(), dtype=np.float16)
    #tfidf_test = np.array(tfidf.transform(test[c]).todense(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        #test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]
        
    del tfidf, tfidf_train
    gc.collect()
    

drop_cols = ["Disaster Name","Wikipedia Link ","Lcat_Scenario","L-CAT ScenGroup","Lcat_id","Description"]
train.drop(drop_cols,axis=1,inplace = True)
X = train.drop("target",axis = 1)
y = train["target"]
feature_names = list(X.columns)

cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class':19,
        'metric':'multi_logloss',
        'max_depth': 16,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
    }  

    
    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        print(tuples[:50])

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    p = list(p)
    p = [int(list(np.where(i == i.max()))[0]) for i in p]
    auc = accuracy_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    #p = model.predict(X_test, num_iteration=model.best_iteration)
    #if len(p_buf) == 0:
     #   p_buf = np.array(p)
    #else:
     #   p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
   # if cnt > 0: # Comment this to run several folds
    #    break
    
    del model
    gc.collect()

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('ACCURACY = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))


labels = list(train["L-CAT ScenGroup"].str.strip().unique())
index = list(range(len(labels)))
zipped = zip(labels,index)
mapping = dict(zipped)

train["target"] = train["L-CAT ScenGroup"].str.strip().map(mapping)

print('Preprocessing text...')
cols = [
   'Description'
]
n_features = [
    400
]
train["Description"] = train["Description"].astype(str)
for c_i, c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i], min_df=3)
    tfidf.fit(train[c])
    tfidf_train = np.array(tfidf.transform(train[c]).todense(), dtype=np.float16)
    #tfidf_test = np.array(tfidf.transform(test[c]).todense(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        #test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]
        
    del tfidf, tfidf_train
    gc.collect()
    

drop_cols = ["Disaster Name","Wikipedia Link ","Lcat_Scenario","L-CAT ScenGroup","Lcat_id","Description"]
train.drop(drop_cols,axis=1,inplace = True)
X = train.drop("target",axis = 1)
y = train["target"]
feature_names = list(X.columns)

cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class':19,
        'metric':'multi_logloss',
        'max_depth': 16,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
    }  

    
    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        print(tuples[:50])

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    p = list(p)
    p = [int(list(np.where(i == i.max()))[0]) for i in p]
    auc = accuracy_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    #p = model.predict(X_test, num_iteration=model.best_iteration)
    #if len(p_buf) == 0:
     #   p_buf = np.array(p)
    #else:
     #   p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
   # if cnt > 0: # Comment this to run several folds
    #    break
    
    del model
    gc.collect()

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))
