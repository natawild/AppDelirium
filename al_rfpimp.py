from sklearn.metrics import r2_score
from rfpimp import permutation_importances
from mlxtend.feature_selection import (
    SequentialFeatureSelector,
    ExhaustiveFeatureSelector,
)
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt


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


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, bootstrap=True)
rf.fit(X_train, y_train)


print(
    "R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}".format(
        rf.score(X_train, y_train), rf.oob_score_, rf.score(X_test, y_test)
    )
)


def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))


perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)

print(perm_imp_rfpimp["Importance"])


"""

# Plot the top 20 feature importance
feature_names_20 = [i[0] for i in feature_importances_ranked[:20]]
y_ticks = np.arange(0, len(feature_names_20))
x_axis = [i[1] for i in feature_importances_ranked[:20]]
plt.figure(figsize = (10, 15))
plt.barh(feature_names_20, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 20)')
#plt.ylabel('Features',fontdict= {'fontsize' : 16})
plt.xlabel('Importance')
plt.show()
"""
