import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    f_classif,
    RFE,
    RFECV,
)
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    fbeta_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from pprint import pprint
from feature_selection_anova import (
    select_features,
    select_k_features,
    print_select_k_features,
)
import os

seed = 100  # so that the result is reproducible


n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)


plt.rcParams["figure.figsize"] = [8.00, 6.50]
plt.rcParams["figure.autolayout"] = True


# Loading dataset
deliriumData = pd.read_csv("./dados_apos_p_processamento.csv")

ax = sns.countplot(deliriumData["Delirium"])
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title("Contagem de categorias da variável ‘Delirium, antes da divisão dos dados")
plt.show()

# TEST.CSV
X = deliriumData.drop("Delirium", axis=1)
y = deliriumData["Delirium"]

X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)
print("Tamanho dos dados de Treino apos ADASYN", X_train.shape, y_train.shape)
print("Tamanho dos dados de Teste", X_test.shape, y_test.shape)


ax = sns.countplot(y_train)
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title("Contagem de categorias da variável ‘Delirium, para os dados de treino")
plt.show()


ax = sns.countplot(y_test)
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title("Contagem de categorias da variável ‘Delirium, para os dados de teste")
plt.show()

print(y_train.shape)


# Classificador Random Forest
model = RandomForestClassifier(
    min_samples_leaf=100,
    n_estimators=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=seed,
    max_features="auto",
)

model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % (accuracy * 100))


report = classification_report(y_test, y_pred)
print(report)

# FInd the feature with best accuracy
results = select_k_features(X_train, y_train)
print("Best Mean Accuracy Random Forest: %.3f" % results.best_score_)
print("Best Config RF: %s" % results.best_params_)


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print("Falsos Posiytivos:", fpr)
print("Verdadeiros Positivos:", tpr)
print("THRESHOLD:", thresholds)
auc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC", auc_score)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
fpr = fp / (fp + tn)

F2 = fbeta_score(y_test, y_pred, beta=2)

print("F2 RF:", F2)

print(precision, recall, fpr)


# Logistic Regression with all columns
model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
report = classification_report(y_test, predicted)
print(report)

# FInd the feature with best accuracy
results = select_k_features(X_train, y_train)
print("Best Mean Accuracy Logistic Regression: %.3f" % results.best_score_)
print("Best Config: %s" % results.best_params_)


# summarize
print("Train LR", X_train.shape, y_train.shape)
print("Test LR", X_test.shape, y_test.shape)
