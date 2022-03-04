import streamlit as st
import numpy as np
import pandas as pd
import utils
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    f_classif,
    RFE,
    RFECV,
    f_regression,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
    KFold,
    RandomizedSearchCV
)

from matplotlib import pyplot
from numpy import mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
import mlxtend
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_recall_curve, 
    auc
)
from numpy import asarray
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
import itertools
from sklearn.utils.fixes import loguniform
import joblib


# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop("Delirium", axis=1)
    y = data["Delirium"]
    return X, y


###############################################################################
#             Train model with all features                                   #
###############################################################################


X, y = load_dataset("./dados_apos_p_processamento.csv")


X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673, stratify=y
)

# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=100)
# rus = RandomUnderSampler(random_state = 32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)


# Create a logistic regression classifier
clf = LogisticRegression(C= 1.0,
 class_weight= None,
 dual= False,
 fit_intercept= True,
 intercept_scaling= 1,
 l1_ratio= None,
 max_iter= 100,
 multi_class= 'ovr',
 n_jobs= None,
 penalty= 'l2',
 random_state= 78787879,
 solver= 'lbfgs',
 tol= 0.0001,
 verbose= 0,
 warm_start= False)

# Train the classifier
clf.fit(X_train, y_train)

###############################################################################
#             Select most important features with  SFM                    #
###############################################################################

# Feature Selection 
sel = SelectFromModel(
    clf,
    threshold=0.09, 
)

sel = sel.fit(X_train, y_train)


###############################################################################
#             Train model with only the most important features               #
###############################################################################


# Create A Data Subset With Only The Most Important Features
X_train_rfc = sel.transform(X_train)
X_test_rfc = sel.transform(X_test)


#create and Train A New Logistic Regression Classifier Using Only Most Important Features
clf.fit(X_train_rfc,y_train)


# save the model to disk
filename = '/Users/user/Documents/GitHub/AppDelirium/final_model.sav'
joblib.dump(clf, filename)
 







