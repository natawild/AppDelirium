import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from sklearn.metrics import roc_auc_score 

from sklearn.metrics import accuracy_score

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import label_binarize

from sklearn.compose import make_column_transformer

from numpy import asarray

import missingno as msno


# Loading dataset
deliriumData = pd.read_csv('./dados_apos_p_processamento.csv')


#TEST.CSV 
X = deliriumData.drop('Delirium',axis=1)
y = deliriumData['Delirium']



seed = 50  # so that the result is reproducible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=45673)


rf = RandomForestClassifier()
lr = LogisticRegression() 


estim = [rf, lr]

sfs = SFS(estimator=rf, 
           k_features='best',
           forward=True, 
           verbose= 1,
           floating=False, 
           scoring='roc_auc',
           cv=5)

pipe = Pipeline([('sfs', sfs), 
                 ('rf', rf)])

param_grid = [
  {'sfs__k_features': [1, 20]}
  ]

gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='roc_auc', 
                  n_jobs=-1, 
                  cv=5,
                  refit=True)

# run gridearch
gs = gs.fit(X_train, y_train)

for i in range(len(gs.cv_results_['params'])):
    print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])



print("Best parameters via GridSearch", gs.best_params_)


plot_sfs(sfs.get_metric_dict(), kind='std_err');


