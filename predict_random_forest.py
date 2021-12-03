import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection

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

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import label_binarize

from sklearn.compose import make_column_transformer

from numpy import asarray

import missingno as msno


# Loading dataset
#deliriumData = pd.read_csv('dados_apos_p_processamento.csv')
#deliriumData = pd.read_csv('dados_under_resample.csv')
deliriumData = pd.read_csv('dados_over_resample.csv')
deliriumData = deliriumData.drop(deliriumData.columns[0], axis=1)


#TEST.CSV 
X = deliriumData.drop('Delirium',axis=1)
y = deliriumData['Delirium']


seed = 100  # so that the result is reproducible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=seed)


rf_classifier = RandomForestClassifier(
                      min_samples_leaf=15,
                      n_estimators=10,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')


rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print(y_pred)

acc = round(accuracy_score(y_test, y_pred),4)*100
print("The accuracy of the model is :", acc)


train_probs = rf_classifier.predict_proba(X_train)[:,1] 
probs = rf_classifier.predict_proba(X_test)[:,1]
train_predictions = rf_classifier.predict(X_train)

print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')

def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    print('Resultados:\n',results)
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']: 
    	resbaseline = round(baseline[metric], 2)
    	resteste = round(results[metric], 2)
    	restrain = round(train_results[metric], 2)
    print("Resultados: ", metric.capitalize(),resbaseline, resteste, restrain)
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 10
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();


x = evaluate_model(y_pred,probs,train_predictions,train_probs)



n_estimators = [int(x) for x in np.linspace(start = 5, stop = 100, num = 10)]
max_features = ['auto', 'log2']  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]   # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 4, 10]    # Minimum number of samples required at each leaf node
bootstrap = [True, False]       # Method of selecting samples for training each tree
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
               'bootstrap': bootstrap}

print('qqqqqqqqqqqqqqqqqqqqq\n', x)

import itertools

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color 
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Let's plot it out
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes = ['0 - Sem Delirium', '1 - Delirium'],
                      title = 'Delirium Confusion Matrix___ este ___')
plt.show()



# Create base model to tune
rf = RandomForestClassifier(oob_score=True)
# Create random search model and fit the data
rf_random = RandomizedSearchCV(
                        estimator = rf,
                        param_distributions = random_grid,
                        n_iter = 100, 
                        cv = 3,
                        verbose=2, random_state=seed, 
                        scoring='roc_auc')

rf_random.fit(X_train, y_train)
rf_random.best_params_

print('BEEEST PARAMS;',rf_random.best_params_)



print(rf_classifier.feature_importances_)
print(f"There are {len(rf_classifier.feature_importances_)} features in total")

feature_importances = list(zip(X_train, rf_classifier.feature_importances_))
print("Aqui as variáveis importantes:\n",feature_importances)
# Then sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, reverse = True)
print("Que será isto? ",feature_importances_ranked)

# Print out the feature and importances
print("AAAAAAAAA, ",feature_importances_ranked)
[print('Feature: {:38} Importance: ', pair) for pair in feature_importances_ranked];

# Plot the top 25 feature importance
feature_names_25 = [i[0] for i in feature_importances_ranked[:25]]
y_ticks = np.arange(0, len(feature_names_25))
x_axis = [i[1] for i in feature_importances_ranked[:25]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_25, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 25)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
#plt.ylabel('Features',fontdict= {'fontsize' : 16})
plt.xlabel('Importance',fontdict= {'fontsize' : 16})
plt.show()

from pprint import pprint
print('Parameters currently in use:\n')
pprint(rf_classifier.get_params())




n_estimators = [int(x) for x in np.linspace(start = 10, stop = 700, num = 50)]
max_features = ['auto', 'log2']  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]   # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 4, 10]    # Minimum number of samples required at each leaf node
bootstrap = [True, False]       # Method of selecting samples for training each tree
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
               'bootstrap': bootstrap}


# Create base model to tune
rf = RandomForestClassifier(oob_score=True)
# Create random search model and fit the data
rf_random = RandomizedSearchCV(
                        estimator = rf,
                        param_distributions = random_grid,
                        n_iter = 100, 
                        cv = 3,
                        verbose=2, random_state=seed, 
                        scoring='roc_auc')

rf_random.fit(X_train, y_train)
rf_random.best_params_

print('BEEEST PARAMS;',rf_random.best_params_)


# To look at nodes and depths of trees use on average
n_nodes = []
max_depths = []
for ind_tree in best_model.estimators_:
       n_nodes.append(ind_tree.tree_.node_count)
       max_depths.append(ind_tree.tree_.max_depth)
print(f'Average number of nodes {int(np.mean(n_nodes))}')   
print(f'Average maximum depth {int(np.mean(max_depths))}')  

# Use the best model after tuning
best_model = rf_random.best_estimator_
pipe_best_model = make_pipeline(col_trans, best_model)
pipe_best_model.fit(X_train, y_train)
y_pred_best_model = pipe_best_model.predict(X_test)



train_rf_predictions = pipe_best_model.predict(X_train)
train_rf_probs = pipe_best_model.predict_proba(X_train)[:, 1]
rf_probs = pipe_best_model.predict_proba(X_test)[:, 1]
# Plot ROC curve and check scores
evaluate_model(y_pred_best_model, rf_probs, train_rf_predictions, train_rf_probs)


# Plot Confusion matrix
plot_confusion_matrix(confusion_matrix(y_test, y_pred_best_model), classes = ['0 - Sem delirium', '1 - Delirium'],
title = 'Exit_status Confusion Matrix___ ou este ')




