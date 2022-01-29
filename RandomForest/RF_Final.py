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
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
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
)
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop("Delirium", axis=1)
    y = data["Delirium"]
    return X, y


X, y = load_dataset("./dados_apos_p_processamento.csv")


X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)

# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
# rus = RandomUnderSampler(random_state = 32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)


# evaluation of a model using features chosen with ... 


# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectKBest(score_func=mutual_info_classif, k=4)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

#Feature names 
features_name = X_train_fs.columns[fs.get_support()]
print(X_train.columns[fs.get_support()])
print(len(features_name))


# fit the model
model = RandomForestClassifier()
model.fit(X_train_fs, y_train)


# evaluate the model
y_pred = model.predict(X_test_fs)


# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

print('Accuracy: \n ', accuracy_score(y_test,y_pred))
print('Recall: \n ', recall_score(y_test,y_pred))
print('Precision: \n ', precision_score(y_test,y_pred))



'''
# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
print('correlated features: ', len(set(corr_features)) )



# step forward feature selection : forward=True;
# step backward feature selection : forward=False;
# 


sfs1 = SFS(RandomForestClassifier(), 
            k_features=(2,30), 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=3, 
           n_jobs=-1)

sfs1 = sfs1.fit(np.array(X_train), y_train)
sbs_cols = X_train.columns[list(sfs1.k_feature_idx_)]
print(sbs_cols)



def show_best_model(support_array, columns, model):
    y_pred = model.predict(X_test.iloc[:, support_array])
    r2 = r2_score(y_test, y_pred)
    n = len(y_pred) #size of test set
    p = len(model.coef_) #number of features
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    print('Adjusted R-squared: %.2f' % adjusted_r2)
    j = 0;
    for i in range(len(support_array)):
        if support_array[i] == True:
            print(columns[i], model.coef_[j])
            j +=1


regr = RandomForestClassifier()

for m in range(1,37):
    selector = RFE(regr, m, step=1) 
    selector.fit(X_train, y_train)
    if m<37:
        show_best_model(selector.support_, x_train.columns, selector.estimator_)


regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Average error: %.2f' %mean(y_test - y_pred))
print('Mean absolute error: %.2f' %mean_absolute_error(y_test, y_pred))
print('Mean absolute error: %.2f' %(mean(abs(y_test - y_pred))))
print("Root mean squared error: %.2f"
      % math.sqrt(mean_squared_error(y_test, y_pred)))
print('percentage absolute error: %.2f' %mean(abs((y_test - y_pred)/y_test)))
print('percentage absolute error: %.2f' %(mean(abs(y_test - y_pred))/mean(y_test)))
print('R-squared: %.2f' % r2_score(y_test, y_pred))



selector.fit(x_train, y_train)
show_best_model(selector.support_, X_train.columns, selector.estimator_)

'''







