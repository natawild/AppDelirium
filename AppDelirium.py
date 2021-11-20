import streamlit as st
import numpy as np
import pandas as pd
import category_encoders as ce

# importing OneHotEncoder
from sklearn.preprocessing import OneHotEncoder()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from dadosSemGasometria import get_user_input_without_gasome
from dadosComGasometria import get_user_input_with_gasome


#Criacao de um título e subtitulo
st.write("""
#Delirium Detection
Detect if someone has delirium using machine learning and python !
""")

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")


#Dados
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Sem Gasometria', 'Com Gasometria')
)

st.write(f"## Dados relativos a {dataset_name} ")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'Random Forest')
)


# FETCH THE DATA AS DATAFRAME 
def get_dataset(name):
    if name == 'Com Gasometria':
        return pd.read_csv('./DeliriumcomGasometria.csv')
    return pd.read_csv('./DeliriumsemGasometria.csv')

#split the data by independent (X) and dependent variables (y)
csvDados = get_dataset(dataset_name)
X = csvDados.iloc[:, 1:-1].values
y = csvDados.iloc[:, -1].values

# configurar um sub titulo
st.subheader('Data Information:')
st.write('Shape of dataset:', X.shape)
st.write('Head of dataset:', X) 
st.write('Head of dataset:', y) 
st.write('number of classes:', len(np.unique(y)))



def CValues (values): 
    out=[]
    for i in range(0, len(values)): 
        out.append(values[i])
        i=i+1
    return out 



#Function to add parametters algorithms 
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 20)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    else:
        values = [100, 10, 1.0, 0.1, 0.01]
        c_values = st.sidebar.selectbox('c_values', CValues(values))
        params['c_values'] = c_values
        max_iter = st.sidebar.slider('max_iter', 1, 1000, 10)
        params['max_iter'] = max_iter
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params['c_values'], max_iter=params['max_iter'])
    else: 
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=3546)
    return clf

clf = get_classifier(classifier_name, params)



#### CLASSIFICATION ####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45673)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)


 
def get_type_of_dataset(name):
    if name == 'Com Gasometria':
        return get_user_input_with_gasome()
    return  get_user_input_without_gasome()




# guardar o input do utilizador numa variavel

user_input = get_type_of_dataset(dataset_name)

# Configurar uma subhead e mostrar aos utilizadores input
st.subheader('User Input:')
st.write(user_input)

# criar e treinar o modelo

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, y_train)

# Mostar as métricas do modelo
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Guardar o modelos preditos numa variavel
prediction = RandomForestClassifier.predict(user_input)

# configurar um subheader e mostrar a classificação
st.subheader('Classification:')
st.write(prediction)



print(rf_classifier.feature_importances_)
print(f" There are {len(rf_classifier.feature_importances_)} features in total")

