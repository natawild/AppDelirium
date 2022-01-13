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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
import itertools
from pprint import pprint 
import matplotlib.pyplot as plt 
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import SelectKBest, RFE, SelectPercentile, f_classif, mutual_info_classif
from sklearn.utils import class_weight
import xgboost as xgb
from collections import Counter


seed = 100  # so that the result is reproducible
plt.rcParams["figure.figsize"] = [7.00, 5.50]
plt.rcParams["figure.autolayout"] = True

# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop('Delirium',axis=1)
    y = data['Delirium']
    return X, y

def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass


def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass



X, y = load_dataset('./dados_apos_p_processamento.csv')

X_train_des, X_test, y_train_des, y_test = train_test_split(X, y, test_size=0.36, random_state=45673)


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste 
rus = ADASYN(random_state = 32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)


unique, count = np.unique(y_train, return_counts=True)
y_train_smote_value_count = { k:v for (k,v) in zip(unique, count)}
print(y_train_smote_value_count)





clf = LogisticRegression().fit(X_train, y_train)

Y_Test_Pred = clf.predict(X_test)

crosstable = pd.crosstab(Y_Test_Pred, y_test, rownames=['Predicted'], colnames=['Actual'])
print(crosstable)

print(generate_model_report(y_test, Y_Test_Pred))
print(generate_auc_roc_curve(clf, X_test))



weights = np.linspace(0.05, 0.95, 20)
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=5
)

grid_result = gsc.fit(X_train, y_train)
print("Best parameters : %s" % grid_result.best_params_)


clf = LogisticRegression(**grid_result.best_params_).fit(X_train, y_train)
Y_Test_Pred = clf.predict(X_test)
cross_table = pd.crosstab(Y_Test_Pred, y_test, rownames=['Predicted'], colnames=['Actual'])
print(cross_table)
print('Regressão Logistica melhores parametros',generate_model_report(y_test, Y_Test_Pred))
print(generate_auc_roc_curve(clf, X_test))


X_train_g, X_test, y_train_g, y_test = train_test_split(X, y, test_size=0.36, random_state=45673)


xgb_model = xgb.XGBClassifier().fit(X_train_g, y_train_g)
y_pred_xgb = xgb_model.predict(X_test)

print('OLA',generate_model_report(y_test, y_pred_xgb))
print(generate_auc_roc_curve(xgb_model, X_test))
crosstable = pd.crosstab(y_pred_xgb, y_test, rownames=['Predicted'], colnames=['Actual'])
print(crosstable)



# Caculating the ratio
counter = Counter(y_train_g)
estimate = counter[0] / counter[1]

# Implementing the model
xgb_model = xgb.XGBClassifier(scale_pos_weight=estimate).fit(X_train_g, y_train_g)

y_pred_xgb_scaled = xgb_model.predict(X_test)


print('OLA XGBClassifier',generate_model_report(y_test, y_pred_xgb_scaled))
print(generate_auc_roc_curve(xgb_model, X_test))
crosstable = pd.crosstab(y_pred_xgb_scaled, y_test, rownames=['Predicted'], colnames=['Actual'])
print(crosstable)

