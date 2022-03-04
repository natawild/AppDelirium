import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
import itertools
from pprint import pprint
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

seed = 100
plt.rcParams["figure.figsize"] = [8.00, 6.50]
plt.rcParams["figure.autolayout"] = True

"""
# Loading dataset
#deliriumData = pd.read_csv('dados_apos_p_processamento.csv')
#deliriumData = pd.read_csv('dados_under_resample.csv')
deliriumData = pd.read_csv('dados_over_resample.csv')
deliriumData = deliriumData.drop(deliriumData.columns[0], axis=1)
"""

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


ax = sns.countplot(y_train_des)
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title("Contagem de categorias da variável ‘Delirium, para os dados de treino")
plt.show()
print(y_train_des.shape)

ax = sns.countplot(y_test)
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title("Contagem de categorias da variável ‘Delirium, para os dados de teste")
plt.show()


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = RandomOverSampler(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)
print(
    "Tamanho dos dados de Treino apos RandomOverSampler", X_train.shape, y_train.shape
)


ax = sns.countplot(y_train)
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title(
    "Contagem de categorias da variável ‘Delirium, para os dados de treino após RandomOverSampler()"
)
plt.show()


rf_classifier = RandomForestClassifier(
    min_samples_leaf=1,
    n_estimators=100,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=seed,
    max_features="auto",
)


fit_rf_class = rf_classifier.fit(X_train, y_train)

y_pred = fit_rf_class.predict(X_test)
print(y_pred)


acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model is :", acc)


c_r = classification_report(y_test, y_pred)
print("Classification report", c_r)

auc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC", auc_score)


# For y_score, ‘The binary case … the scores must be the scores of the class with the greater label’. That is why we need to get label 1 instead of label 0.
train_probs = rf_classifier.predict_proba(X_train)[:, 1]
probs = rf_classifier.predict_proba(X_test)[:, 1]
train_predictions = rf_classifier.predict(X_train)

print(f"Train ROC AUC Score: {roc_auc_score(y_train, train_probs)}")
print(f"Test ROC AUC  Score: {roc_auc_score(y_test, probs)}")


def evaluate_model(y_pred, probs, train_predictions, train_probs):
    baseline = {}
    baseline["recall"] = recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline["precision"] = precision_score(y_test, [1 for _ in range(len(y_test))])
    baseline["roc"] = 0.5
    results = {}
    results["recall"] = recall_score(y_test, y_pred)
    results["precision"] = precision_score(y_test, y_pred)
    results["roc"] = roc_auc_score(y_test, probs)
    print("Resultados:\n", results)
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
    plt.rcParams["font.size"] = 12
    # Plot both curves
    plt.plot(base_fpr, base_tpr, "b", label="Classificador aleatóreo")
    plt.plot(model_fpr, model_tpr, "r", label="model")
    plt.legend()
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("ROC Curves")
    plt.show()


evaluate_model(y_pred, probs, train_predictions, train_probs)
plt.show()


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix 6666", cmap=plt.cm.Greens
):  # can change color
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=20)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=10)
    plt.yticks(tick_marks, classes, size=10)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=15,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("True label", size=15)
    plt.xlabel("Predicted label", size=15)


# Let's plot it out
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm,
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Delirium Confusion Matrix - parametros padrão",
)
plt.show()


print(rf_classifier.feature_importances_)
print(f"There are {len(rf_classifier.feature_importances_)} features in total")

feature_importances = list(zip(X_train, rf_classifier.feature_importances_))
print("Aqui as variáveis importantes:\n", feature_importances)

# Then sort the feature importances by most important first
feature_importances_ranked = sorted(
    feature_importances, key=lambda x: x[1], reverse=True
)

# Print out the feature and importances
[print("Feature: {:38} Importance: ", pair) for pair in feature_importances_ranked]


# Plot the top 20 feature importance
feature_names_20 = [i[0] for i in feature_importances_ranked[:20]]
y_ticks = np.arange(0, len(feature_names_20))
x_axis = [i[1] for i in feature_importances_ranked[:20]]
plt.figure(figsize=(10, 15))
plt.barh(feature_names_20, x_axis)  # horizontal barplot
plt.title("Random Forest Feature Importance (Top 20)")
# plt.ylabel('Features',fontdict= {'fontsize' : 16})
plt.xlabel("Importance")
plt.show()


# Tune the hyperparameters with RandomSearchCV

print("Parameters currently in use:\n")
pprint(rf_classifier.get_params())

# create a grid of parameters for the model to randomly pick and train, hence the name Random Search

n_estimators = [int(x) for x in np.arange(start=2, stop=110, step=1)]
max_features = ["auto", "log2", "sqrt"]  # Number of features to consider at every split
max_depth = [
    int(x) for x in np.arange(start=1, stop=80, step=1)
]  # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [
    int(x) for x in np.arange(start=2, stop=20, step=1)
]  # Minimum number of samples required to split a node
min_samples_leaf = [
    int(x) for x in np.arange(start=1, stop=20, step=1)
]  # Minimum number of samples required at each leaf node
criterion = ["gini", "entropy"]
bootstrap = [True]  # Method of selecting samples for training each tree
oob_score = [True]

"""
def func_oob (bootstrap): 
    oob_score = [] 
    random_grid = {}
    if bootstrap == True: oob_score = True
    if bootstrap == False: oob_score = False
    return oob_score
"""


random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "criterion": criterion,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_leaf_nodes": [None] + list(np.arange(5, 200, 5).astype(int)),
    "bootstrap": bootstrap,
    "oob_score": oob_score,
}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=seed,
    scoring="roc_auc",
)

rf_random.fit(X_train, y_train)
best_params = rf_random.best_params_
print("Best Params\n")
pprint(best_params)
best_model = rf_random.best_estimator_
print("AQUI", best_params["max_leaf_nodes"])
pprint(rf_random)
rf_test = RandomForestClassifier(
    min_samples_leaf=best_params["min_samples_leaf"],
    max_leaf_nodes=best_params["max_leaf_nodes"],
    criterion=best_params["criterion"],
    n_estimators=best_params["n_estimators"],
    bootstrap=best_params["bootstrap"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    max_features=best_params["max_features"],
)
rf_test.fit(X_train, y_train)

y_pred = rf_test.predict(X_test)
train_rf_predictions = rf_test.predict(X_train)

print("BBB\n")
pprint(y_pred)

train_rf_probs = rf_test.predict_proba(X_train)[:, 1]
rf_probs = rf_test.predict_proba(X_test)[:, 1]
# Plot ROC curve and check scores
evaluate_model(y_pred, rf_probs, train_rf_predictions, train_rf_probs)
plt.show()

plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Nao", "1 - Sim"],
    title="Delirium Confusion Matrix Apos escolha de parametros",
)
plt.show()


# Supervised transformation based on random forests
rf = RandomForestClassifier()
rf_lr = LogisticRegression()
rf.fit(X_train, y_train)
rf_lr.fit(X_train, y_train)

y_pred_rf_lr = rf_lr.predict_proba(X_test)[:, 1]
fpr_rf_lr, tpr_rf_lr, _ = roc_curve(y_test, y_pred_rf_lr)


plt.plot(fpr_rf_lr, tpr_rf_lr, label="RF + LR")
plt.show()
"""
n_nodes = []
max_depths = []
for ind_tree in best_model.estimators_:
       n_nodes.append(ind_tree.tree_.node_count)
       max_depths.append(ind_tree.tree_.max_depth)
print(f'Average number of nodes {int(np.mean(n_nodes))}') 
print(f'Average maximum depth {int(np.mean(max_depths))}')  

# Use the best model after tuning
pipe_best_model = make_pipeline(X_test, best_model)
pipe_best_model.fit(X_train, y_train)
y_pred_best_model = pipe_best_model.predict(X_test)


# Create base model to tune
rf = RandomForestClassifier(oob_score=True)

# Create random search model and fit the data
rf_random = RandomizedSearchCV(
                        estimator = rf,
                        param_distributions = random_grid,
                        n_iter = 10, 
                        cv = 3,
                        verbose=2, random_state=seed, 
                        scoring='roc_auc')

rf_random.fit(X_train, y_train)
rf_random.best_params_


#Use the best model after tuning
best_model = rf_random.best_estimator_
print('BEST ', best_model)
pipe_best_model = make_pipeline(X_train, best_model)
pipe_best_model.fit(X_train, y_train)
y_pred_best_model = pipe_best_model.predict(X_test)




# To look at nodes and depths of trees use on average
n_nodes = []
max_depths = []
for ind_tree in best_model.estimators_:
       n_nodes.append(ind_tree.tree_.node_count)
       max_depths.append(ind_tree.tree_.max_depth)
print(f'Average number of nodes {int(np.mean(n_nodes))}')   
print(f'Average maximum depth {int(np.mean(max_depths))}')  



train_rf_predictions = pipe_best_model.predict(X_train)
train_rf_probs = pipe_best_model.predict_proba(X_train)[:, 1]
rf_probs = pipe_best_model.predict_proba(X_test)[:, 1]
# Plot ROC curve and check scores
evaluate_model(y_pred_best_model, rf_probs, train_rf_predictions, train_rf_probs)


# Plot Confusion matrix
plot_confusion_matrix(confusion_matrix(y_test, y_pred_best_model), classes = ['0 - Sem delirium', '1 - Delirium'],
title = 'Exit_status Confusion Matrix___ ou este ')

"""
