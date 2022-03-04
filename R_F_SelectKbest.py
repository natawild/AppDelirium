from sklearn.metrics import r2_score
from rfpimp import permutation_importances
from mlxtend.feature_selection import (
    SequentialFeatureSelector,
    ExhaustiveFeatureSelector,
)
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from numpy import mean, std


deliriumData = pd.read_csv("./dados_apos_p_processamento.csv")

# TEST.CSV
X = deliriumData.drop("Delirium", axis=1)
y = deliriumData["Delirium"]

X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)

print(y_test.head(20))

# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)

print("Tamanho dos dados de Treino apos ADASYN", X_train.shape, y_train.shape)

columnsSize = len(X.columns)


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    return scores


for i in range(1, columnsSize):
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=i)
    X_new = rfe.fit_transform(X_train, y_train)
    # print(X_new)
    print(rfe.get_feature_names_out())
    feature_importance = rfe.estimator_.feature_importances_
    # feature_importance = rfe.feature_importances_
    # feature_importances = pd.DataFrame(feature_importance,
    #                                index = X_train.columns,
    #                                columns=['importance']).sort_values('importance', ascending = False)
    print("Colunas importancia: ", feature_importance)
    scores = evaluate_model(rfe, X_test, y_test)
    print("> %.3f (%.3f)" % (mean(scores), std(scores)))


# support = pipeline.named_steps['rfe_feature_selection'].support_
# feature_names = pipeline.named_steps['features'].get_feature_names()
# np.array(feature_names)[support]
