import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from sklearn.metrics import roc_auc_score 
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import mlxtend
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection



# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv(filename)
	# split into input (X) and output (y) variables
	X = data.drop('Delirium',axis=1)
	y = data['Delirium']
	return X, y

X, y = load_dataset('./dados_apos_p_processamento.csv')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=45673)

# define number of features to evaluate
num_features = [i+1 for i in range(X.shape[1])]



# get a list of models to evaluate
def get_models():
	models = dict()
	for i in num_features:
		rfe = RFECV(estimator=LogisticRegression())
		model = LogisticRegression()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models
'''
# get a list of models to evaluate
def get_models():
	models = dict()
	for i in num_features:
		rfe = RFE(estimator=LogisticRegression(), n_features_to_select=i)
		model = LogisticRegression()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

'''
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores



# get the models to evaluate
models = get_models()


# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


