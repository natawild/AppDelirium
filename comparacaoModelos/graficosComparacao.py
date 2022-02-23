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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
import itertools
from sklearn.utils.fixes import loguniform

# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop("Delirium", axis=1)
    y = data["Delirium"]
    return X, y


###############################################################################
#             Train model Logistic Regresion with all features                                   #
###############################################################################


X, y = load_dataset("./dados_apos_p_processamento.csv")
#X, y = load_dataset("./resnovos3.csv")


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

# Apply The Full Featured Classifier To The Test Data
y_pred_all = clf.predict(X_test)
yhat = clf.predict_proba(X_test)
yhat_positive_all_lr = yhat[:, 1]

# View The Accuracy
acc = round(accuracy_score(y_test, y_pred_all), 4) * 100
print("The accuracy of the model Logistic Regression  is:\n", acc)

recall = round(recall_score(y_test, y_pred_all), 4) * 100
print("The recall of the model Logistic Regression is:\n", recall)

precision = round(precision_score(y_test, y_pred_all), 4) * 100
print("The precision of the model Logistic Regression is:\n", precision)

f1 = round(f1_score(y_test, y_pred_all), 4) * 100
print("The f1 of the model Logistic Regression is:\n", f1)

c_r = classification_report(y_test, y_pred_all)
print("Classification report Logistic Regression\n", c_r)

auc_score = roc_auc_score(y_test, yhat_positive_all_lr)
print("ROC AUC Logistic Regression \n", auc_score)

matrix = confusion_matrix(y_test, y_pred_all)
print(matrix)

#--------------------precision_recall_curve---------------

# predict probabilities
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('Logistic Regression: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))



###############################################################################
#             Select most important features with  SFM                    #
###############################################################################
# Feature Selection 
sel = SelectFromModel(
    clf,
    threshold=0.09, 
)
print("Parametros SelectFromModel\n \n \n ",sel)

sel = sel.fit(X_train, y_train)

###############################################################################
#             Train model with only the most important features               #
###############################################################################

# Create A Data Subset With Only The Most Important Features
X_train_rlc = sel.transform(X_train)
X_test_rlc = sel.transform(X_test)


#create and Train A New Logistic Regression Classifier Using Only Most Important Features
clf_rl_sel = LogisticRegression()
clf_rl_sel.fit(X_train_rlc,y_train)

#predict results 
y_pred_sel = clf_rl_sel.predict(X_test_rlc)
print("###########################\n",y_pred_sel)

#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred_sel), 4) * 100
print("The accuracy of the model is:\n", acc)

# predict probabilities
yhat = clf_rl_sel.predict_proba(X_test_rlc)

# retrieve the probabilities for the positive class
yhat_positive_rl_sel = yhat[:, 1]

# calculate and print AUROC
roc_auc = roc_auc_score(y_test, yhat_positive_rl_sel)
print('ROC CURVE ---- : %.3f' % roc_auc)

recall = round(recall_score(y_test, y_pred_sel), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred_sel), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred_sel), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred_sel)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, yhat_positive_rl_sel)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred_sel)
print(matrix)


lr_precision_rl, lr_recall_rl, _ = precision_recall_curve(y_test, yhat_positive_rl_sel)
lr_f1, lr_auc = f1_score(y_test, y_pred_sel), auc(lr_recall_rl, lr_precision_rl)

# summarize scores
print('Regressão L: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))



#--------------------------PLOT FEATURE IMPORTANCES------------------------

features_name = sel.get_feature_names_out()
#print('indices das features selecionadas: ', sfs.k_feature_idx_)
print('Length Features selected: \n',len(features_name))
print("Features selected:\n ", features_name)

print('intercept \n', clf.intercept_[0])
print('classes \n', clf.classes_)
print('value of coef_:\n',clf.coef_[0])

# Print slope and intercept
print('Intercept (Beta 0): ', clf.intercept_)
print('Slope (Beta 1): ', clf.coef_)
'''
print(pd.DataFrame({'coeff': clf.coef_[0]}, 
             index=X_train_rfc.columns))
'''


################################## Random Forest ##############################################################################

# Create a random forest classifier
clf_random = RandomForestClassifier(random_state=11463247)

# Train the classifier
clf_random.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(X_train.columns, clf_random.feature_importances_):
    print(feature)


# Apply The Full Featured Classifier To The Test Data
y_pred = clf_random.predict(X_test)
yhat = clf_random.predict_proba(X_test)
yhat_positive = yhat[:, 1]

# View The Accuracy
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model Random Forest is:\n", acc)

recall = round(recall_score(y_test, y_pred), 4) * 100
print("The recall of the model Random Forest is:\n", recall)

precision = round(precision_score(y_test, y_pred), 4) * 100
print("The precision of the model Random Forest is:\n", precision)

f1 = round(f1_score(y_test, y_pred), 4) * 100
print("The f1 of the model Random Forest is:\n", f1)


c_r = classification_report(y_test, y_pred)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, yhat_positive)
print("ROC AUC - Random Forest \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)


# predict probabilities
lr_probs = clf_random.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf_random.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('Random Forest sem seleção: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))


###############################################################################
#             Select most important features with RFECV                      #
###############################################################################

sel = RFECV(RandomForestClassifier(random_state=11463247))
sel.fit(X_train,y_train)

###############################################################################
#             Train model with only the most important features               #
###############################################################################

# Create A Data Subset With Only The Most Important Features
X_train_rfc = sel.transform(X_train)
X_test_rfc = sel.transform(X_test)

#Train A New Random Forest Classifier Using Only Most Important Features
# Create a new random forest classifier for the most important features
clf_random_most = RandomForestClassifier()
clf_random_most.fit(X_train_rfc,y_train)

##predict results 
y_pred_rf_sel = clf_random_most.predict(X_test_rfc)

# predict probabilities
yhat = clf_random_most.predict_proba(X_test_rfc)

# retrieve the probabilities for the positive class
yhat_positive = yhat[:, 1]

#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred_rf_sel), 4) * 100
print("The accuracy of the model Random Forest after selected is:\n", acc)

# calculate and print AUROC
roc_auc = roc_auc_score(y_test, yhat_positive)
print('AUROC Random Forest after selected: %.3f' % roc_auc)

recall = round(recall_score(y_test, y_pred_rf_sel), 4) * 100
print("The recall of the model Random Forest after selected is:\n", recall)

precision = round(precision_score(y_test, y_pred_rf_sel), 4) * 100
print("The precision of the model Random Forest after selected is:\n", precision)

f1 = round(f1_score(y_test, y_pred_rf_sel), 4) * 100
print("The f1 of the model Random Forest after selected is:\n", f1)

c_r = classification_report(y_test, y_pred_rf_sel)
print("Classification report Random Forest after selected\n", c_r)

matrix = confusion_matrix(y_test, y_pred_rf_sel)
print(matrix)


# predict probabilities
lr_probs = clf_random_most.predict_proba(X_test_rfc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf_random_most.predict(X_test_rfc)
lr_precision_rf, lr_recall_rf, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall_rf, lr_precision_rf)

# summarize scores
print('Random Forest após selecçao: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))


################################################################################################################################

summary = pd.DataFrame(
    data={
        "labels": ["Acurácia", "FVP", "FNV", "F1", "ROC_AUC", "E-S_AUC"],
        "Regressão Logistica (SFM)": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, yhat_positive),
            auc(lr_recall_rl, lr_precision_rl)
        ],
        "Random Forest (RFECV)": [
            accuracy_score(y_test, y_pred_rf_sel),
            precision_score(y_test, y_pred_rf_sel),
            recall_score(y_test, y_pred_rf_sel),
            f1_score(y_test, y_pred_rf_sel),
            roc_auc_score(y_test, y_pred_rf_sel),
            auc(lr_recall_rf, lr_precision_rf)
        ],
        
    }
).set_index("labels")
summary.index.name = None

print(summary)



summary1 = pd.DataFrame(
    data={
        "labels": ["Acurácia", "FVP", "FNV", "F1", "ROC_AUC", "E-S_AUC"],
        "Regressão Logística (SFM)": [
            0.8471,
            0.60,
            0.6774,
            0.6364,
            0.8333,
            0.582
        ],
        "Random Forest (RFECV)": [
            0.7898,
            0.40,
            0.5385,
            0.459,
            0.789,
            0.508
        ],
        
    }
).set_index("labels")
summary1.index.name = None

print(summary1)

fig, ax = plt.subplots(figsize=(12, 6))
summary1.plot.bar(ax=ax)
ax.legend(bbox_to_anchor=(1, 1), frameon=False)
ax.grid(False)
ax.set_title("Gráfico de comparação das métricas entre RF e RL")
plt.xticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(12, 6))
summary.plot.bar(ax=ax)
ax.legend(bbox_to_anchor=(1, 1), frameon=False)
ax.grid(False)
ax.set_title("Gráfico de comparação das métricas entre o modelo RF e RL")
plt.xticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()


