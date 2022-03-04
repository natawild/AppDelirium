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


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Greens
):  # can change color
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=13)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=12,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("Valor Real", size=12)
    plt.xlabel("Valor Previsto", size=12)
    plt.show()



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
rus = ADASYN(random_state=32)
# rus = RandomUnderSampler(random_state = 32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)


# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(X_train.columns, clf.feature_importances_):
    print(feature)


# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)
yhat = clf.predict_proba(X_test)
yhat_positive = yhat[:, 1]

# View The Accuracy
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model is:\n", acc)

recall = round(recall_score(y_test, y_pred), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred), 4) * 100
print("The f1 of the model is:\n", f1)


c_r = classification_report(y_test, y_pred)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, yhat_positive)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Look at parameters used by current model
print('Parameters currently in use:\n')
pprint(clf.get_params())

# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Matriz de confusão sem seleção de variáveis (RF_SFS_FLOAT_FORW)",
)


# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# plot roc curve
plt.plot(fpr, tpr, color='red', label='AUC ' + str(auc_score))
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
# axis labels
plt.xlabel('1 - Especificidade (FFP)')
plt.ylabel('Sensibilidade (FVP)')
plt.title('Curva ROC')
# show the legend
plt.legend()
plt.show()


# predict probabilities
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('RF: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, label='AUC ' + str(lr_auc))
# axis labels
pyplot.title('Curva Especificidade-Sensibilidade')
pyplot.xlabel('Sensibilidade')
pyplot.ylabel('Especificidade')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


###############################################################################
#             Select most important features with RF stepwise forward                      #
###############################################################################

sel = SequentialFeatureSelector(
    clf,
    #k_features=1,
    #k_features=(1, 38),
    k_features="best",
    forward=True,
    floating=True,
    scoring="f1",
    cv=3,
    n_jobs=-1,
)
sel = sel.fit(X_train, y_train)
features_name = X_train.columns[list(sel.k_feature_idx_)]
#print('indices das features selecionadas: ', sfs.k_feature_idx_)
print('Length Features selected: \n',len(features_name))
print("Features selected:\n ", features_name)


print("Dictonary with metrics")
#http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-4-plotting-the-results
pd.DataFrame.from_dict(sel.get_metric_dict()).T.to_csv("RF_SFS_FLOAT_FORW_dict.csv")

###############################################################################
#             Train model with only the most important features               #
###############################################################################

# Create A Data Subset With Only The Most Important Features
X_train_rfc = sel.transform(X_train)
X_test_rfc = sel.transform(X_test)


#Train A New Random Forest Classifier Using Only Most Important Features
# Create a new random forest classifier for the most important features
clf = RandomForestClassifier()
clf.fit(X_train_rfc,y_train)

#predict results 
y_pred = clf.predict(X_test_rfc)


# predict probabilities
yhat = clf.predict_proba(X_test_rfc)


# retrieve the probabilities for the positive class
yhat_positive = yhat[:, 1]


#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model is:\n", acc)


# calculate and print AUROC
roc_auc = roc_auc_score(y_test, yhat_positive)
print('AUROC: %.3f' % roc_auc)

recall = round(recall_score(y_test, y_pred), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred)
print("Classification report\n", c_r)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# plot roc curve
plt.plot(fpr, tpr, color='red', label='AUC ' + str(roc_auc))
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
# axis labels
plt.xlabel('1 - Especificidade (FFP)')
plt.ylabel('Sensibilidade (FVP)')
plt.title('Curva ROC das variáveis selecionadas (RF_SFS_FLOAT_FORW)')
# show the legend
plt.legend()
plt.show()

    
# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Matriz de confusão após seleção de variáveis (RF_SFS_FLOAT_FORW)",
)



# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(clf.get_params())


# predict probabilities
lr_probs = clf.predict_proba(X_test_rfc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf.predict(X_test_rfc)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('RF: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, label='AUC ' + str(lr_auc))
# axis labels
pyplot.title('Curva Especificidade-Sensibilidade após seleção variáveis')
pyplot.xlabel('Sensibilidade')
pyplot.ylabel('Especificidade')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



###############################################################################
#             Tunning model                                                   #
###############################################################################

seed=100

# create a grid of parameters for the model to randomly pick and train, hence the name Random Search
n_estimators = [100, 200, 500]
max_features = ['auto']  # Number of features to consider at every split
max_depth = [10,13, 15, 20, 50, 95, 100]   # Maximum number of levels in tree
min_samples_split = [2, 12, 20, 30]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 10 , 20]   # Minimum number of samples required at each leaf node
criterion = ['gini', 'entropy']
max_leaf_nodes = [50, 100, 120]
bootstrap = [True]  # Method of selecting samples for training each tree
oob_score = [True]


'''
# create a grid of parameters for the model to randomly pick and train, hence the name Random Search
n_estimators = [int(x) for x in np.arange(start = 2, stop = 120, step = 12)]
max_features = ['auto', 'log2', 'sqrt']  # Number of features to consider at every split
max_depth = [int(x) for x in np.arange(start = 1, stop = 100, step = 10)]   # Maximum number of levels in tree
min_samples_split = [int(x) for x in np.arange(start = 2, stop = 20, step = 10)]  # Minimum number of samples required to split a node
min_samples_leaf = [int(x) for x in np.arange(start = 1, stop = 20, step = 10)]   # Minimum number of samples required at each leaf node
criterion = ['gini', 'entropy']
max_leaf_nodes = [int(x) for x in np.arange(start = 1, stop = 100, step = 10)]
bootstrap = [True]  # Method of selecting samples for training each tree
oob_score = [True]
'''

random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": max_leaf_nodes,
    "criterion":criterion,
    "max_leaf_nodes":max_leaf_nodes,
    "bootstrap": bootstrap,
    "oob_score":oob_score
}


# Create random search model and fit the data
rf_random = RandomizedSearchCV(
    estimator=clf,
    param_distributions=random_grid,
    n_iter=50,
    #cv=RepeatedStratifiedKFold(),
    cv = 3,
    verbose=0,
    random_state=seed,
    scoring="f1",
)

'''
#-------------------------------------------GridSearchCV--------------------------------------------------#

grid = GridSearchCV(estimator=clf,
                    param_grid=rf_random,
                    scoring='roc_auc',
                    cv = RepeatedStratifiedKFold(), 
                    verbose=0,
                    n_jobs=-1)

'''

grid_result = rf_random.fit(X_train_rfc, y_train)

# print winning set of hyperparameters
print("Best set of hyperparameters \n ",grid_result.best_estimator_.get_params())
print('Best params: \n', grid_result.best_params_)



# Use the best model after tuning
best_model = grid_result.best_estimator_

best_model.fit(X_train_rfc, y_train)

y_pred_best_model = best_model.predict(X_test_rfc)
train_rf_predictions = best_model.predict(X_train_rfc)
train_rf_probs = best_model.predict_proba(X_train_rfc)[:, 1]
rf_probs = best_model.predict_proba(X_test_rfc)[:, 1]


#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred_best_model), 4) * 100
print("The accuracy of the model is:\n", acc)

recall = round(recall_score(y_test, y_pred_best_model), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred_best_model), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred_best_model), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred_best_model)
print("Classification report\n", c_r)

matrix = confusion_matrix(y_test, y_pred_best_model)
print(matrix)

auc_score= roc_auc_score(y_test, rf_probs)
print("AUC SCORE:", auc_score)


# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
# plot roc curve
plt.plot(fpr, tpr, color='red', label='AUC ' + str(auc_score))
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
# axis labels
plt.xlabel('1 - Especificidade (FFP)')
plt.ylabel('Sensibilidade (FVP)')
plt.title('Curva ROC após afinação (RF_SFS_FLOAT_FORW)')
# show the legend
plt.legend()
plt.show()


# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred_best_model),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Matriz de confusão após tuning (RF_SFS_FLOAT_FORW)",
)


# To look at nodes and depths of trees use on average
n_nodes = []
max_depths = []
for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
print(f"Average number of nodes {int(np.mean(n_nodes))}")
print(f"Average maximum depth {int(np.mean(max_depths))}")


auc_score= roc_auc_score(y_test, rf_probs)
print("AUC SCORE:", auc_score)


# predict probabilities
lr_probs = best_model.predict_proba(X_test_rfc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = best_model.predict(X_test_rfc)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('RF: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, label='AUC ' + str(lr_auc))
# axis labels
pyplot.title('Curva Especificidade-Sensibilidade após afinação')
pyplot.xlabel('Sensibilidade')
pyplot.ylabel('Especificidade')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



