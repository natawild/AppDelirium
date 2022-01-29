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
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=20,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.grid(None)
    plt.tight_layout()import streamlit as st
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
import itertools



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


# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(X_train.columns, clf.feature_importances_):
    print(feature)


# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

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

auc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Plot Confusion matrix
utils.plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="without tuning Delirium Confusion Matrix",
)


# Feature Selection by feature importance of random forest classifier 

sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_train,y_train)

#return true or false if feature are selected or not 
features_support =sel.get_support()
features_name = X_train.columns[features_support]
print('Length Features selected: \n',len(features_name))
print("Features selected:\n ", features_name)

print(X_train.columns)




feature_importance = sel.estimator_.feature_importances_
features_importance_pair = utils.get_index_value(feature_importance,features_support,X_train.columns)
print(features_importance_pair)


# Create A Data Subset With Only The Most Important Features
X_train_rfc = sel.transform(X_train)
X_test_rfc = sel.transform(X_test)

print('X TRAIN',X_train_rfc.shape)
print('Y TRAIN',y_train.shape)
print('X Test',X_test_rfc.shape)
print('Y TEST',y_test.shape)

#Train A New Random Forest Classifier Using Only Most Important Features
# Create a new random forest classifier for the most important features
clf = RandomForestClassifier()
clf.fit(X_train_rfc,y_train)

#predict results 
y_pred = clf.predict(X_test_rfc)

#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model is:\n", acc)


# predict probabilities
yhat = clf.predict_proba(X_test_rfc)

# retrieve the probabilities for the positive class
yhat_positive = yhat[:, 1]
print("yhat_positive: ", yhat_positive)

# calculate and print AUROC
roc_auc = roc_auc_score(y_test, yhat_positive)
print('AUROC: %.3f' % roc_auc)

# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# plot roc curve
plt.plot(fpr, tpr, marker='.', label='AUC ' + str(roc_auc))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
plt.show()
    
# Plot Confusion matrix
utils.plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="After selected Features Delirium Confusion Matrix",
)

recall = round(recall_score(y_test, y_pred), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(clf.get_params())



###############################################################################
#             4c. Feature Selection: Selecting relevant features              #
###############################################################################

seed=100

# create a grid of parameters for the model to randomly pick and train, hence the name Random Search
n_estimators = [int(x) for x in np.arange(start = 2, stop = 250, step = 1)]
max_features = ['auto', 'log2', 'sqrt']  # Number of features to consider at every split
max_depth = [int(x) for x in np.arange(start = 1, stop = 100, step = 1)]   # Maximum number of levels in tree
min_samples_split = [int(x) for x in np.arange(start = 2, stop = 20, step = 1)]  # Minimum number of samples required to split a node
min_samples_leaf = [int(x) for x in np.arange(start = 1, stop = 20, step = 1)]   # Minimum number of samples required at each leaf node
criterion = ['gini', 'entropy']
max_leaf_nodes = [int(x) for x in np.arange(start = 1, stop = 100, step = 1)]
bootstrap = [True]  # Method of selecting samples for training each tree


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
}

# Create base model to tune
rf = RandomForestClassifier(oob_score=True)

# Create random search model and fit the data
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=seed,
    scoring="roc_auc",
)

rf_random.fit(X_train_rfc, y_train)

print('Best params: ', rf_random.best_params_)
best_model = rf_random.best_estimator_


# To look at nodes and depths of trees use on average
n_nodes = []
max_depths = []
for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
print(f"Average number of nodes {int(np.mean(n_nodes))}")
print(f"Average maximum depth {int(np.mean(max_depths))}")

# Use the best model after tuning
best_model = rf_random.best_estimator_
print(best_model)


#pipe_best_model = make_pipeline(col_trans, best_model)
best_model.fit(X_train_rfc, y_train)
y_pred_best_model = best_model.predict(X_test_rfc)


train_rf_predictions = best_model.predict(X_train_rfc)
train_rf_probs = best_model.predict_proba(X_train_rfc)[:, 1]
rf_probs = best_model.predict_proba(X_test_rfc)[:, 1]
# Plot ROC curve and check scores
utils.evaluate_model(y_pred_best_model, rf_probs, train_rf_predictions, train_rf_probs)


# Plot Confusion matrix
utils.plot_confusion_matrix(
    confusion_matrix(y_test, y_pred_best_model),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="After Delirium Confusion Matrix",
)


    plt.ylabel("True label", size=18)
    plt.xlabel("Predicted label", size=18)
    plt.show()


def evaluate_model(y_pred, probs, train_predictions, train_probs):
    baseline = {}
    baseline["recall"] = recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline["precision"] = precision_score(y_test, [1 for _ in range(len(y_test))])
    baseline["roc"] = 0.5
    results = {}
    results["recall"] = recall_score(y_test, y_pred)
    results["precision"] = precision_score(y_test, y_pred)
    results["roc"] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results["recall"] = recall_score(y_train, train_predictions)
    train_results["precision"] = precision_score(y_train, train_predictions)
    train_results["roc"] = roc_auc_score(y_train, train_probs)
    for metric in ["recall", "precision", "roc"]:
        resbaseline = round(baseline[metric], 2)
        resteste = round(results[metric], 2)
        restrain = round(train_results[metric], 2)
    print("Resultados: ", metric.capitalize(), resbaseline, resteste, restrain)
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, "b", label="baseline")
    plt.plot(model_fpr, model_tpr, "r", label="model")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.show()


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


# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(X_train.columns, clf.feature_importances_):
    print(feature)


# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

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

auc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="without tuning Delirium Confusion Matrix",
)


# Feature Selection by feature importance of random forest classifier 

sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_train,y_train)

#return true or false if feature are selected or not 
features_support =sel.get_support()
features_name = X_train.columns[features_support]
print('Length Features selected: \n',len(features_name))
print("Features selected:\n ", features_name)

print(X_train.columns)


#Feature importance 
def get_index_value ( arr, features_support, column_names ):
    features = []
    for index, value in enumerate(arr): 
        if features_support[index]:
            pair = [ arr[index], column_names[index]]
            features.append(pair)
    return features


feature_importance = sel.estimator_.feature_importances_
features_importance_pair = get_index_value(feature_importance,features_support,X_train.columns)
print(features_importance_pair)


# Create A Data Subset With Only The Most Important Features
X_train_rfc = sel.transform(X_train)
X_test_rfc = sel.transform(X_test)

print('X TRAIN',X_train_rfc.shape)
print('Y TRAIN',y_train.shape)
print('X Test',X_test_rfc.shape)
print('Y TEST',y_test.shape)

#Train A New Random Forest Classifier Using Only Most Important Features
# Create a new random forest classifier for the most important features
clf = RandomForestClassifier()
clf.fit(X_train_rfc,y_train)

#predict results 
y_pred = clf.predict(X_test_rfc)

#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model is:\n", acc)


# predict probabilities
yhat = clf.predict_proba(X_test_rfc)

# retrieve the probabilities for the positive class
yhat_positive = yhat[:, 1]
print("yhat_positive: ", yhat_positive)

# calculate and print AUROC
roc_auc = roc_auc_score(y_test, yhat_positive)
print('AUROC: %.3f' % roc_auc)

# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# plot roc curve
plt.plot(fpr, tpr, marker='.', label='AUC ' + str(roc_auc))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
plt.show()
    
# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="After selected Features Delirium Confusion Matrix",
)

recall = round(recall_score(y_test, y_pred), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(clf.get_params())



###############################################################################
#             4c. Feature Selection: Selecting relevant features              #
###############################################################################



###############################################################################
#                          Tuning of selected features             #
###############################################################################

seed=100

# create a grid of parameters for the model to randomly pick and train, hence the name Random Search
n_estimators = [int(x) for x in np.arange(start = 2, stop = 250, step = 1)]
max_features = ['auto', 'log2', 'sqrt']  # Number of features to consider at every split
max_depth = [int(x) for x in np.arange(start = 1, stop = 100, step = 1)]   # Maximum number of levels in tree
min_samples_split = [int(x) for x in np.arange(start = 2, stop = 20, step = 1)]  # Minimum number of samples required to split a node
min_samples_leaf = [int(x) for x in np.arange(start = 1, stop = 20, step = 1)]   # Minimum number of samples required at each leaf node
criterion = ['gini', 'entropy']
max_leaf_nodes = [int(x) for x in np.arange(start = 1, stop = 100, step = 1)]
bootstrap = [True]  # Method of selecting samples for training each tree


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
}

# Create base model to tune
rf = RandomForestClassifier(oob_score=True)

# Create random search model and fit the data
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=seed,
    scoring="roc_auc",
)

rf_random.fit(X_train_rfc, y_train)

print('Best params: ', rf_random.best_params_)
best_model = rf_random.best_estimator_


# To look at nodes and depths of trees use on average
n_nodes = []
max_depths = []
for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
print(f"Average number of nodes {int(np.mean(n_nodes))}")
print(f"Average maximum depth {int(np.mean(max_depths))}")

# Use the best model after tuning
best_model = rf_random.best_estimator_
print(best_model)


#pipe_best_model = make_pipeline(col_trans, best_model)
best_model.fit(X_train_rfc, y_train)
y_pred_best_model = best_model.predict(X_test_rfc)


train_rf_predictions = best_model.predict(X_train_rfc)
train_rf_probs = best_model.predict_proba(X_train_rfc)[:, 1]
rf_probs = best_model.predict_proba(X_test_rfc)[:, 1]
# Plot ROC curve and check scores
evaluate_model(y_pred_best_model, rf_probs, train_rf_predictions, train_rf_probs)


# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred_best_model),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="After Delirium Confusion Matrix",
)

