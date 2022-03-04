import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from dadosSemGasometria import get_user_input_without_gasome
from dadosComGasometria import get_user_input_with_gasome

import missingno as msno


# Loading dataset
deliriumData = pd.read_csv("./dados_apos_p_processamento.csv")

# Viewing data
deliriumData.head()
deliriumData.info()

print("Olá aqui vao as keys", deliriumData.keys())
print(deliriumData.Idade)
print(deliriumData.Glicose)

# X = pd.DataFrame(deliriumData)
# y = deliriumData.target

# TEST.CSV
X = deliriumData.drop("Delirium", axis=1)
y = deliriumData["Delirium"]

print("aqui vai o X", X.head())
print("aqui vai o y", y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45673
)
X_train.shape
X_test.shape


# a list with all missing value formats
missing_value_formats = ["n.a.", "?", "NA", "n/a", "na", "--"]
deliriumDatas = pd.read_csv("./DadosDelirium.csv", na_values=missing_value_formats)
print(deliriumDatas.isnull().head(20))

print("Dados em falta", deliriumData.isnull().sum())


# check is there any missing values in dataframe as a whole
missingData = deliriumData.isnull()
print(missingData)

sns.heatmap(deliriumData.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.show()

# Check if there is any missing values across each column
deliriumData.isnull().any()


msno.matrix(deliriumData)


plt.figure(figsize=(10, 6))
sns.heatmap(
    deliriumData.isna().transpose(), cmap="YlGnBu", cbar_kws={"label": "Missing Data"}
)
plt.savefig("visualizing_missing_data_with_heatmap_Seaborn_Python.png", dpi=100)

# plt.show()


"""
#Step Foward Feature Selection (SFS)
sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
	k_features = 2,
	forward = True, 
	floating = False, 
	verbose = 2,
	scoring = 'accuracy',
	cv = 4,
	n_jobs = -1
	).fit(X_train, y_train)


# Which features?
feat_cols = list(sfs.k_feature_idx_)
print(feat_cols)


print(sfs.k_features_names_)
print(sfs.k_score)


"""
