import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
import itertools
from pprint import pprint
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    SelectPercentile,
    f_classif,
    mutual_info_classif,
)


deliriumData = pd.read_csv("./dados_apos_p_processamento.csv")

# TEST.CSV
X = deliriumData.drop("Delirium", axis=1)
y = deliriumData["Delirium"]

X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)

print(y_test.head(20))

# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)

print("Tamanho dos dados de Treino apos ADASYN", X_train.shape, y_train.shape)


rf = RandomForestClassifier()
lr = LogisticRegression()

estim = [rf, lr]

sfs = SFS(
    estimator=rf,
    k_features="best",
    forward=True,
    verbose=1,
    floating=False,
    scoring="roc_auc",
    cv=5,
)

pipe = Pipeline([("sfs", sfs), ("rf", rf)])

param_grid = [{"sfs__k_features": [1, 20]}]

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    n_jobs=-1,
    cv=5,
    refit=True,
)

# run gridSearch
gs = gs.fit(X_train, y_train)

for i in range(len(gs.cv_results_["params"])):
    print(
        gs.cv_results_["params"][i], "test acc.:", gs.cv_results_["mean_test_score"][i]
    )


print("Best parameters via GridSearch", gs.best_params_)
plot_sfs(sfs.get_metric_dict(), kind="std_err")
plt.show()


"""
pipeline = Pipeline([
					('model', Lasso)
					])

search = GridSearchCV(pipeline,
					{'model':np.arange(0.1,3,0.1)},
					cv = 5,
					scoring = 'neg_mean_squared_error',
					verbose = 3, 
					error_score='raise')

search.fit(X_train,y_train)

search.best_params_ 

coef = search.best_estimator_[1].coef
print(np.array(features)[coef!=0])
"""
