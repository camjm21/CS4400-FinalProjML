#!/usr/bin/env python
# coding: utf-8

# # CS 4400 Final Project

# Imports:

# In[1]:


import py_entitymatching as em
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Read in data:

# In[2]:


ltable = em.read_csv_metadata("ltable.csv", key='id')
rtable = em.read_csv_metadata("rtable.csv", key='id')
train = pd.read_csv("train.csv")
train['_id'] = train.index
cols = train.columns.tolist()
cols = cols[-1:] + cols[:-1]
train = train[cols]


# Setting up the training datframe:

# In[3]:


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

# In[4]:


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

# In[5]:


ab = em.AttrEquivalenceBlocker()
c_blocked = ab.block_tables(ltable, rtable, 'brand', 'brand', ['id', 'title', 'category', 'brand', 'modelno', 'price'], ['id', 'title', 'category', 'brand', 'modelno', 'price'] )


# Setting Attribute Correspondences:

# In[6]:


atypes2 = em.get_attr_types(rtable)
atypes2['title'] = 'str_bt_5w_10w'
blk = em.get_attr_corres(ltable, rtable)
blk['corres'].pop(0)
blk['corres'].pop(1)
blk['corres'].pop(1)
blk['corres'].pop(1)
blk['corres'].pop(1)


# Generating features:

# In[7]:


tok = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()
feature_table = em.get_features(ltable, rtable, em.get_attr_types(ltable), atypes2, blk, tok, sim)


# Creating a set of learning-based matchers:

# In[8]:


dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
nb = em.NBMatcher(name='NaiveBayes')
xg = em.XGBoostMatcher(name = 'XGBoost')
rf2 = RandomForestClassifier(class_weight="balanced", random_state=0)


# Converting the training set to feature vectors:

# In[9]:


tr_feat = em.extract_feature_vecs(tr, feature_table = feature_table)
tr_feat2 = tr_feat.iloc[:, 3:]
tr_feat['label'] = train['label']


# Concluding that Random Forest is the best learner:

# In[10]:


result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=tr_feat, 
        exclude_attrs=['_id', 'id_l', 'id_r', 'label'],
        k=5,
        target_attr='label', metric_to_select_matcher='f1', random_state=0)
result['cv_stats']


# Extracting data from feature vector dataframe into a list of lists:

# In[11]:


tr_list = tr_feat2.values.tolist()


# Fitting the Random Forest learner to the training set of data:

# In[12]:


rf2.fit(tr_list, training_labels[:4500])


# Extracting feature vectors from the blocked candidate set:

# In[13]:


cand_feats = em.extract_feature_vecs(c_blocked, feature_table = feature_table)


# Gathering data from cand_set and predicting the set using Random Forest:

# In[15]:


cand_feats2 = cand_feats.iloc[:, 3:]
cand_list = cand_feats2.values.tolist()
p = rf2.predict(cand_list)


# Output of matches to csv file:

# In[16]:


matching_pairs = c_blocked.loc[p == 1, ["ltable_id", "rtable_id"]]
matching_pairs = list(map(tuple, matching_pairs.values))


# In[17]:


matching_pairs_in_training = training_df.loc[training_labels == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))


# In[18]:


pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)


# In[ ]:




