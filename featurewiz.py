import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    mutual_info_classif,
    VarianceThreshold,
)

plt.rcParams["figure.figsize"] = [7.00, 5.50]
plt.rcParams["figure.autolayout"] = True


deliriumData = pd.read_csv("./dados_apos_p_processamento.csv")


# TEST.CSV
X = deliriumData.drop("Delirium", axis=1)
y = deliriumData["Delirium"]


X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)

print("Tamanho dos dados de Treino apos ADASYN", X_train.shape, y_train.shape)


correlated_features = set()
correlation_matrix = X.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            print(correlated_features.add(colname))


rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring="accuracy")
rfecv.fit(X_train, y_train)


# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring shows the proportion of correct classifications

min_features_to_select = 2  # Minimum number of features to consider

rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=StratifiedKFold(2),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)


rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(rfecv.cv_results_) + min_features_to_select),
    rfecv.cv_results_,
)
plt.show()
