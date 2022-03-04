import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
import category_encoders as ce

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# define data
data = asarray([["red"], ["green"], ["blue"]])
print(data)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)

csvDados = pd.read_csv("./dados.csv")
# print(csvDados.dtypes)


dados = pd.read_csv("./dadosCategoricos.csv")
dummies = pd.get_dummies(dados)
print("Dumies a caminho\n", dummies)
print("Colunas :\n", dummies.columns)


novo_df = pd.read_csv(
    "./dadosCategoricos.csv",
    usecols=[
        "Genero",
        "GrupoDiagn",
        "Rosuvastatina",
        "Atorvastatina",
        "Pravastatina",
        "Sinvastatina",
        "Fluvastatina",
        "Alprazolam",
        "Captopril",
        "Desloratadine",
        "Digoxin",
        "Dipyridamole",
        "Furosemide",
        "Fluvoxamine",
        "Haloperidol",
        "Hydrocortisone",
        "Iloperidone",
        "Morphine",
        "Nifedipine",
        "Ranitidine",
        "Risperidone",
        "Trazodone",
        "Venlafaxine",
        "Warfarin",
        "Amitriptyline",
        "Hydroxyzine",
        "Paroxetine",
        "Quetiapine",
        "Scopolamine",
        "Trihexyphenidyl",
        "Clonidine",
        "Tramadol",
        "Mexazolam",
        "Trospium",
        "Alcoolico",
        "ResultDelirium",
    ],
)

print("Dados só com as variáveis categóricas\n", novo_df.head())

# unique values in each columns
for x in novo_df.columns:
    # printing unique values
    print("Valores unicos:\n", x, ":", len(novo_df[x].unique()))


# finding the categories
print(
    "Categorias do grupo de diagnóstico:\n",
    novo_df.GrupoDiagn.value_counts().sort_values(ascending=False).head(10),
)
cat_grupo_diag = [
    x
    for x in novo_df.GrupoDiagn.value_counts()
    .sort_values(ascending=False)
    .head(10)
    .index
]

# make binary of labels
for label in cat_grupo_diag:
    novo_df[label] = np.where(novo_df["GrupoDiagn"] == label, 1, 0)
    novo_df[["GrupoDiagn"] + cat_grupo_diag]

print("Novo data\n", novo_df)

# print("Dataset com dummies\n",res)
# print("Tipo de dados do res\n", res.dtypes)


# print("Valores do X:",X)
# print("valores do y:\n",y)


mapeamentoMedicamentos = {"Presente": 1, "Ausente": 0}
mapeamentoBool = {1: True, 0: False}
mapeamentoDelirium = {"Delirium": 1, "Sem delirium": 0}
mapeamentoGrupoDiagnostico = {
    "Neurologico": 0,
    "Cardiovascular": 1,
    "Gastrointestinal": 2,
    "Respiratório": 3,
    "Genitourinário": 4,
    "Musculoesquelético": 5,
    "Toxicidade de Drogas": 6,
    "Outro": 7,
    "Hemato-Oncológico": 8,
}

mapeamentoGrupoDiagnostco = {
    0: "Neurologico",
    1: "Cardiovascular",
    2: "Gastrointestinal",
    3: "Respiratório",
    4: "Genitourinário",
    5: "Musculoesquelético",
    6: "Toxicidade de Drogas",
    7: "Outro",
    8: "HematoOncológico",
}

# print(csvDados.isnull().sum())
# print(csvDados.head())


# print(X_train.dtypes)

csvDados["Genero"] = csvDados["Genero"].map(mapeamentoBool)
csvDados["Rosuvastatina"] = csvDados["Rosuvastatina"].map(mapeamentoBool)
csvDados["Atorvastatina"] = csvDados["Atorvastatina"].map(mapeamentoBool)
csvDados["Pravastatina"] = csvDados["Pravastatina"].map(mapeamentoBool)
csvDados["Sinvastatina"] = csvDados["Sinvastatina"].map(mapeamentoBool)
csvDados["Fluvastatina"] = csvDados["Fluvastatina"].map(mapeamentoBool)
csvDados["Alprazolam"] = csvDados["Alprazolam"].map(mapeamentoBool)
csvDados["Captopril"] = csvDados["Captopril"].map(mapeamentoBool)
csvDados["Desloratadine"] = csvDados["Desloratadine"].map(mapeamentoBool)
csvDados["Digoxin"] = csvDados["Digoxin"].map(mapeamentoBool)
csvDados["Dipyridamole"] = csvDados["Dipyridamole"].map(mapeamentoBool)
csvDados["Furosemide"] = csvDados["Furosemide"].map(mapeamentoBool)
csvDados["Fluvoxamine"] = csvDados["Fluvoxamine"].map(mapeamentoBool)
csvDados["Haloperidol"] = csvDados["Haloperidol"].map(mapeamentoBool)
csvDados["Hydrocortisone"] = csvDados["Hydrocortisone"].map(mapeamentoBool)
csvDados["Iloperidone"] = csvDados["Iloperidone"].map(mapeamentoBool)
csvDados["Morphine"] = csvDados["Morphine"].map(mapeamentoBool)
csvDados["Nifedipine"] = csvDados["Nifedipine"].map(mapeamentoBool)
csvDados["Prednisone"] = csvDados["Prednisone"].map(mapeamentoBool)
csvDados["Ranitidine"] = csvDados["Ranitidine"].map(mapeamentoBool)
csvDados["Risperidone"] = csvDados["Risperidone"].map(mapeamentoBool)
csvDados["Trazodone"] = csvDados["Trazodone"].map(mapeamentoBool)
csvDados["Venlafaxine"] = csvDados["Venlafaxine"].map(mapeamentoBool)
csvDados["Warfarin"] = csvDados["Warfarin"].map(mapeamentoBool)
csvDados["Amitriptyline"] = csvDados["Amitriptyline"].map(mapeamentoBool)
csvDados["Hydroxyzine"] = csvDados["Hydroxyzine"].map(mapeamentoBool)
csvDados["Paroxetine"] = csvDados["Paroxetine"].map(mapeamentoBool)
csvDados["Quetiapine"] = csvDados["Quetiapine"].map(mapeamentoBool)
csvDados["Scopolamine"] = csvDados["Scopolamine"].map(mapeamentoBool)
csvDados["Trihexyphenidyl"] = csvDados["Trihexyphenidyl"].map(mapeamentoBool)
csvDados["Clonidine"] = csvDados["Clonidine"].map(mapeamentoBool)
csvDados["Tramadol"] = csvDados["Tramadol"].map(mapeamentoBool)
csvDados["Mexazolam"] = csvDados["Mexazolam"].map(mapeamentoBool)
csvDados["Trospium"] = csvDados["Trospium"].map(mapeamentoBool)
csvDados["Alcoolico"] = csvDados["Alcoolico"].map(mapeamentoBool)
csvDados["ResultDelirium"] = csvDados["ResultDelirium"].map(mapeamentoBool)
csvDados["Obito"] = csvDados["Obito"].map(mapeamentoBool)
csvDados["Diazepam"] = csvDados["Diazepam"].map(mapeamentoBool)
csvDados["Sertralina"] = csvDados["Sertralina"].map(mapeamentoBool)
csvDados["Paliperidone"] = csvDados["Paliperidone"].map(mapeamentoBool)
csvDados["Lorazepam"] = csvDados["Lorazepam"].map(mapeamentoBool)
csvDados["GrupoDiagn"] = csvDados["GrupoDiagn"].map(mapeamentoGrupoDiagnostco)


X = csvDados.iloc[:, 1:-1].values
y = csvDados.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45673
)

print(csvDados.head())


print("AQUI\n", csvDados.dtypes)

# Create a list of categorical variables
# features_to_encode = list(X_train.select_dtypes(include = ['object']).columns)
print(X_train)


print(features_to_encode)

# Create a constructor to handle categorical features for us
col_trans = make_column_transformer(
    (OneHotEncoder(), features_to_encode), remainder="passthrough"
)


plt.figure(figsize=(80, 40))
plot_tree(
    rf_best.estimators_[5],
    feature_names=X.columns,
    class_names=["Delirium", "Sem Delirium"],
    filled=True,
)
plt.show()


plt.rcParams["figure.figsize"] = [20.0, 7.0]
plt.rcParams.update(
    {
        "font.size": 22,
    }
)

sns.set_palette("viridis")
sns.set_style("white")
sns.set_context("talk", font_scale=0.8)

dados = pd.read_csv("./dadosCategoricos.csv")
dummies = pd.get_dummies(dados)
print("Dummies:\n", dummies.dtypes)

csvDados = pd.read_csv("./dados.csv")
# print(csvDados.dtypes)
print(csvDados.info())

# check for missing values
def missing_values_table(csvDados):
    # Total missing values
    mis_val = csvDados.isnull().sum()
    print(mis_val)

    # Percentage of missing values
    mis_val_percent = 100 * csvDados.isnull().sum() / len(csvDados)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )

    # Print some summary information
    print(
        "O conjunto de dados tem " + str(csvDados.shape[1]) + " colunas.\n"
        "Existem "
        + str(mis_val_table_ren_columns.shape[0])
        + " colunas que contém valores omissos"
    )

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


print(missing_values_table(csvDados))


# A expressão iloc[:, :-1] indica que queremos todas as linhas e todas as colunas exceto a
# última ([linhas, colunas]) e
# iloc[:, -1] indica que queremos apenas a coluna no ultimo indice.
X = csvDados.iloc[:, 1:-1].values
y = csvDados.iloc[:, -1].values

print(X.shape)

fig, ax = plt.subplots()

"""
# using seaborns countplot to show distribution of questions in dataset
fig, ax = plt.subplots()
g = sns.countplot(csvDados, palette='viridis')
g.set_xticklabels(['0', '1'])
g.set_yticklabels([])
"""

# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = "{:.0f}".format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel("")
plt.ylabel("")
plt.title("Distribution of Target", fontsize=30)
plt.tick_params(axis="x", which="major", labelsize=15)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45673
)

print(csvDados.head())

print("AQUI\n", csvDados.dtypes)
