#!/usr/bin/env python
# coding: utf-8

# # CS 4400 Final Project

# Imports:

# In[301]:


import py_entitymatching as em
import pandas as pd
import os
from os.path import join
import numpy as np
import Levenshtein as lev
from sklearn.ensemble import RandomForestClassifier


# Read in data and set metadata:

# In[302]:


ltable = em.read_csv_metadata("ltable.csv", key='id')
rtable = em.read_csv_metadata("rtable.csv", key='id')
train = pd.read_csv("train.csv")
train['_id'] = train.index
cols = train.columns.tolist()
cols = cols[-1:] + cols[:-1]
train = train[cols]
em.set_key(train, '_id')
em.set_key(ltable, 'id')
em.set_key(rtable, 'id')
em.set_ltable(train, ltable)
em.set_rtable(train, rtable)
em.set_fk_ltable(train, 'ltable_id')
em.set_fk_rtable(train, 'rtable_id')


# Setting up the training datframe:

# In[303]:


def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_df['label'] = train['label']
training_df['_id'] = training_df.index
cols = training_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
training_df = training_df[cols]
training_labels = train.label.values


# Providing metadata for training dataframe and splitting training_df into a train and test set:

# In[304]:


em.set_key(training_df, '_id')
em.set_ltable(training_df, ltable)
em.set_rtable(training_df, rtable)
em.set_fk_ltable(training_df, 'id_l')
em.set_fk_rtable(training_df, 'id_r')
em.get_fk_ltable(training_df)
trte = em.split_train_test(training_df, train_proportion = 0.9)
tr = trte['train']
te = trte['test']


# Instantiate a blocker object and block based off brand:

# In[305]:


ab = em.AttrEquivalenceBlocker()
cand_set = ab.block_tables(ltable, rtable, 'brand', 'brand', ['id', 'title', 'category', 'brand', 'modelno', 'price'], ['id', 'title', 'category', 'brand', 'modelno', 'price'] )


# Setting Attribute Correspondences:

# In[306]:


atypes1 = em.get_attr_types(ltable)
atypes2 = em.get_attr_types(rtable)
block_c = em.get_attr_corres(ltable, rtable)
block_c['corres'].pop(0)
block_c['corres'].pop(1)
atypes2['title'] = 'str_bt_5w_10w'
block = block_c
block['corres'].pop(1)
block['corres'].pop(1)
block['corres'].pop(1)


# Generating features:

# In[307]:


tok = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()
feature_table = em.get_features(ltable, rtable, atypes1, atypes2, block, tok, sim)


# Creating a set of learning-based matchers:

# In[308]:


dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
nb = em.NBMatcher(name='NaiveBayes')
xg = em.XGBoostMatcher(name = 'XGBoost')
rf2 = RandomForestClassifier(class_weight="balanced", random_state=0)


# Converting the training set to feature vectors:

# In[309]:



H = em.extract_feature_vecs(tr, feature_table = feature_table)
H2 = H.iloc[:, 3:]
H['label'] = train['label']


# Concluding that Random Forest is the best learner:

# In[310]:


result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H, 
        exclude_attrs=['_id', 'id_l', 'id_r', 'label'],
        k=5,
        target_attr='label', metric_to_select_matcher='f1', random_state=0)
result['cv_stats']


# Extracting data from feature vector dataframe into a list of lists:

# In[311]:



lol = H2.values.tolist()


# Fitting the Random Forest learner to the training set of data:

# In[312]:


rf2.fit(lol, training_labels[:4500])


# Extracting feature vectors from the cand_set:

# In[313]:


C = em.extract_feature_vecs(cand_set, feature_table = feature_table)


# Gathering data from cand_set and predicting the set using Random Forest:

# In[314]:


C2 = C.iloc[:, 3:]
lol3 = C2.values.tolist()
p = rf2.predict(lol3)


# Output of matches to csv file:

# In[315]:


matching_pairs = cand_set.loc[p == 1, ["ltable_id", "rtable_id"]]
matching_pairs = list(map(tuple, matching_pairs.values))


# In[316]:



matching_pairs_in_training = training_df.loc[training_labels == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))


# In[317]:


pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)


# In[ ]:




