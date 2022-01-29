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


seed = 100  # so that the result is reproducible


def setupChart(plt):
    plt.rcParams["figure.figsize"] = [7.00, 5.50]
    plt.rcParams["figure.autolayout"] = True


# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop("Delirium", axis=1)
    y = data["Delirium"]
    return X, y


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise"
    )
    return scores


X, y = load_dataset("./dados_apos_p_processamento.csv")

X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)


# create a grid of parameters for the SelectKbest to randomly pick and train


"""
A biblioteca de máquinas de scikit-learn fornece uma implementação do teste f da ANOVA na função f_classif(). 
Esta função pode ser utilizada numa estratégia de selecção de características, 
tal como a selecção das características mais relevantes (maiores valores) através da classe SelectKBest.
"""

# f_classif --> F-statistic for each feature
# p_values --> P-values associated with the F-statistic


"""
Em vez de adivinhar, podemos testar sistematicamente uma gama de diferentes números de características selecionadas
 e descobrir qual deles resulta no modelo com melhor desempenho. A isto chama-se uma pesquisa em grelha, onde o argumento 
 k para a classe SelectKBest pode ser afinado.
"""

# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the pipeline to evaluate
model = RandomForestClassifier(n_jobs=-1)
fs = SelectKBest(score_func=f_classif)
pipeline = Pipeline(steps=[("anova", fs), ("rf", model)])


"""
Note-se que a grelha é um dicionário de parâmetros a valores a pesquisar, e dado que estamos a utilizar um Pipeline, 
podemos aceder ao objecto SelectKBest através do nome que lhe demos, 'anova', e depois o nome do parâmetro 'k', separado 
por dois sublinhados, ou 'anova__k'.
"""
# define the grid
grid = dict()
grid["anova__k"] = [i + 1 for i in range(X_train.shape[1])]


# define the grid search
search = GridSearchCV(pipeline, grid, scoring="accuracy", n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X_train, y_train)

# summarize best
print("Best Mean Accuracy: %.3f" % results.best_score_)
print("Best Config: %s" % results.best_params_)


df_scores = pd.DataFrame(
    {
        "features": X_train.columns,
        "f_classif": results.scores_,
        "pValue": results.pvalues_,
    }
)
print(df_scores)


# plot the scores
pyplot.bar([i for i in range(len(results.scores_))], results.scores_)
pyplot.show()

# plot the pValues
pyplot.bar([i for i in range(len(results.pvalues_))], results.pvalues_)
pyplot.show()


# Print the names of the most important features


# Modeling With Selected Features


# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = fs.transform(X_train)
X_important_test = fs.transform(X_test)

print("Forma das selecionadas", X_important_train.shape)

# Train A New Random Forest Classifier Using Only Most Important Features

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier()

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)


# Compare The Accuracy Of Our Full Feature Classifier To Our Limited Feature Classifier
# Apply The Full Featured Classifier To The Test Data
y_pred = classifier.predict(X_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
accuracy_score(y_test, y_pred)
print("View The Accuracy Of Our Full Feature", accuracy_score(y_test, y_pred))


# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
accuracy_score(y_test, y_important_pred)
print(
    "View The Accuracy Of Our Limited Feature ",
    accuracy_score(y_test, y_important_pred),
)


# define number of features to evaluate
num_features = [i + 1 for i in range(X_train.shape[1])]
# enumerate each number of features
results = list()
for k in num_features:
    # create pipeline
    model = RandomForestClassifier()
    fs = SelectKBest(score_func=f_classif, k=k)
    pipeline = Pipeline(steps=[("anova", fs), ("rf", model)])
    # evaluate the model
    scores = evaluate_model(pipeline, X, y)
    results.append(scores)
    # summarize the results
    print(">%d %.3f (%.3f)" % (k, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
