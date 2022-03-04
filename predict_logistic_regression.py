import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
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
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    SelectPercentile,
    f_classif,
    mutual_info_classif,
)


seed = 100  # so that the result is reproducible
plt.rcParams["figure.figsize"] = [7.00, 5.50]
plt.rcParams["figure.autolayout"] = True


"""
# Loading dataset
#deliriumData = pd.read_csv('dados_apos_p_processamento.csv')
#deliriumData = pd.read_csv('dados_under_resample.csv')
deliriumData = pd.read_csv('train_data_adasyn.csv')
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
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)
print("Tamanho dos dados de Treino apos ADASYN", X_train.shape, y_train.shape)


ax = sns.countplot(y_train)
for p in ax.patches:
    ax.annotate(
        "{:.1f}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01)
    )
ax.set_title(
    "Contagem de categorias da variável ‘Delirium, para os dados de treino após ADASYN()"
)
plt.show()


# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)

# get importance
importance = model.coef_[0]
print(importance)
# summarize feature importance
for i, v in enumerate(importance):
    print("Feature: %0d, Score: %.5f" % (i, v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Logistic Regression Importance")
# plt.ylabel('Features',fontdict= {'fontsize' : 16})
plt.xlabel("Importance")
plt.show()


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


rf_classifier = LogisticRegression()


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


# create a grid of parameters for the model to randomly pick and train, hence the name Random Search
max_iter = [int(x) for x in np.arange(start=2, stop=200, step=1)]
penalty = [
    "l1",
    "l2",
    "elasticnet",
    "none",
]  # Number of features to consider at every split
solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]

"""
def func_oob (bootstrap): 
    oob_score = [] 
    random_grid = {}
    if bootstrap == True: oob_score = True
    if bootstrap == False: oob_score = False
    return oob_score
"""

random_grid = {"solver": solver, "max_iter": max_iter, "penalty": penalty}

rf = LogisticRegression()
rf_random = RandomizedSearchCV(
    estimator=rf, param_distributions=random_grid, cv=5, verbose=2, random_state=seed
)

rf_random.fit(X_train, y_train)


best_params = rf_random.best_params_
rf_test = LogisticRegression(
    solver=best_params["solver"],
    max_iter=best_params["max_iter"],
    penalty=best_params["penalty"],
)


rf_test.fit(X_train, y_train)
y_pred = rf_test.predict(X_test)

print("Resultados da previsão:\n")
print(y_pred)


print(confusion_matrix(y_test, y_pred), ": is the confusion matrix \n")
