from numpy import loadtxt
from numpy import unique
import pandas as pd
from numpy import arange
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from numpy import std
from numpy import absolute
from numpy import mean
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

'''
var = ['Proveniência', 'Local_SU', 'Idade','Genero','Interna_Dias', 'GrupoDiagn', 'SIRS', 'Glicose', 'Sodio', 'Ureia',
       'Creatinina', 'PCR', 'pH', 'Ca_ionizado', 'pCO2', 'pO2', 'HCO3','Antidislipidemicos', 'Antipsicóticos', 'Antidepressores',
       'Anti-hipertensores', 'Anti-histaminicos', 'Ansioliticos',
       'Analgésicos ', 'Anticoagulantes ', 'Corticosteroides',
       'Antiespasmódicos', 'Antiparkinsónico', 'Cardiotonico', 'Antiacido ',
       'Geniturinario', 'Obito', 'Alcoolico', 'ResultDelirium']
     

#Renomear os nomes das colunas: 
dadosCategoricos.rename(columns={'Name':'Nome','Idade','Sexo', 'ETC'})

'''
# types
pd.set_option('display.max_columns', 500)

dadosCategoricos = pd.read_csv('./dadosCategoricosAgrupados.csv')
#print('Onde esta isto?',dadosCategoricos.info())
#print(dadosCategoricos.head(20))
#print(dadosCategoricos.keys)
#print(dadosCategoricos.info())
#print(dadosCategoricos.describe())

features_to_dummies = dadosCategoricos[['Genero','Antidislipidemicos', 'Antipsicóticos', 'Antidepressores',
       												'Anti-hipertensores', 'Anti-histaminicos', 'Ansioliticos',
       												'Analgésicos ', 'Anticoagulantes ', 'Corticosteroides',
      												'Antiespasmódicos', 'Antiparkinsónico', 'Cardiotonico', 'Antiacido ',
      												'Geniturinario', 'Obito', 'Alcoolico', 'ResultDelirium']]


features_to_one_hot_encoder = dadosCategoricos[['Proveniência', 'Local_SU', 'GrupoDiagn']]


features_to_normalize = dadosCategoricos[['Idade','Interna_Dias', 'SIRS', 'Glicose', 'Sodio', 'Ureia',
       'Creatinina', 'PCR', 'pH', 'Ca_ionizado', 'pCO2', 'pO2', 'HCO3']]

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], drop_first=True)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 


def encode_label_encoder(original_dataframe, feature_to_encode):
    label_encoder = OneHotEncoder().fit_transform(original_dataframe[[feature_to_encode]])
    df = pd.DataFrame(label_encoder, columns=[feature_to_encode])
    res = original_dataframe.drop([feature_to_encode], axis=1)
    res = pd.concat([res, df], axis=1)
    return(res) 

def normalize_data(original_dataframe, feature_to_encode):
    normalize = MinMaxScaler().fit_transform(original_dataframe[[feature_to_encode]])
    df = pd.DataFrame(normalize, columns=[feature_to_encode])
    res = original_dataframe.drop([feature_to_encode], axis=1)
    res = pd.concat([res, df], axis=1)
    return(res) 


for feature in features_to_one_hot_encoder:
    dadosCategoricos = encode_label_encoder(dadosCategoricos, feature)

for feature in features_to_normalize:
    dadosCategoricos = normalize_data(dadosCategoricos, feature)


for feature in features_to_dummies:
    dadosCategoricos = encode_and_bind(dadosCategoricos, feature)



#print("Após categorização\n",dadosCategoricos)

#print("Desccrição\n",dadosCategoricos.describe())

dadosCategoricos.rename(columns={'ResultDelirium_Sem delirium':'ResultDelirium'}, inplace=True)

# types
pd.set_option('display.max_columns', 500)
print("FEATURE\n", dadosCategoricos)


last_ix = len(dadosCategoricos.columns)-1
#print(last_ix)
X = dadosCategoricos.drop(['ResultDelirium'], axis=1)
y = dadosCategoricos['ResultDelirium']

#print(X.shape, y.shape)
# determine categorical and numerical features


#reverse = list(LabelEncoder.inverse_transform(col_transform['le']))
#print(col_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


model = LogisticRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
report = classification_report(y_test, predicted)
print(report)


    df = pd.DataFrame(ohe, columns=[feature_to_encode])
    res = pd.concat([original_dataframe, ohe], axis=1)
    res = original_dataframe.drop([feature_to_encode], axis=1)
    return(res) 