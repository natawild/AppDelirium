import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE



# Loading dataset
deliriumData = pd.read_csv('./dados_apos_p_processamento.csv')

#TEST.CSV 
X = deliriumData.drop('Delirium',axis=1)
y = deliriumData['Delirium']

# Random Undersampler
rus = RandomUnderSampler(random_state = 32)
X_ros_res, y_ros_res = rus.fit_resample(X, y)

dadosUnderResample = pd.concat([X_ros_res, y_ros_res], axis=1)
dadosUnderResample.to_csv('dados_under_resample.csv')

rus = RandomOverSampler(random_state = 32)
X_ros_res, y_ros_res = rus.fit_resample(X, y)

dadosOverResample = pd.concat([X_ros_res, y_ros_res], axis=1)
dadosOverResample.to_csv('dados_over_resample.csv')


ax = sns.countplot(x="Delirium", data=dadosOverResample)
ax.legend('Sim','Não')

plt.show()

sns.countplot(deliriumData['Delirium'])
plt.show()



seed = 50  # so that the result is reproducible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=45673)


#model built using all features 
seed= 50 
model = RandomForestClassifier(
                      min_samples_leaf=100,
                      n_estimators=100,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')

model.fit(X_train, y_train)
# evaluate the model
y_pred = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print('FPR:',fpr)
print('TPR:',tpr)
print('THRESHOLD:',thresholds)
auc_score = roc_auc_score(y_test, y_pred)
print('ROC AUC',auc_score)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = tp/(tp + fp)
recall = tp/(tp + fn)
fpr = fp/(fp + tn)

F2 = fbeta_score(y_test, y_pred, beta=2)

print('F2:',F2)

print(precision,recall, fpr)



# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)