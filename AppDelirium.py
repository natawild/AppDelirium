import streamlit as st
import numpy as np
import pandas as pd

# importing OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from dadosSemGasometria import get_user_input_without_gasome
from dadosComGasometria import get_user_input_with_gasome

import joblib


#Page headers
st.set_page_config(
    page_title='Delirium Detection', 
    page_icon=None, 
    layout="wide", 
    initial_sidebar_state="expanded", 
    menu_items=None
    )

#Page Title
st.title('Delirium Detection')


# load the model from disk
clf = joblib.load('final_model.sav')

#Criacao de um título e subtitulo
st.write("""
#Delirium Detection
Detect if someone has delirium using machine learning and python !
""")

filters_container = st.container()

fcol1, fcol2, fcol3 = filters_container.columns(3)
fcol1.subheader("A wide column with a chart")
fcol2.subheader("A wide column with a chart")
fcol3.subheader("A wide column with a chart")




'''
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
'''

# configurar um sub titulo
#st.subheader('Data Information:')
#st.write('Shape of dataset:', X.shape)
#st.write('Head of dataset:', X) 
#st.write('Head of dataset:', y) 
#st.write('number of classes:', len(np.unique(y)))

#### CLASSIFICATION ####


#calculate predict
#y_pred = clf.predict(user_input)

# guardar o input do utilizador numa variavel

user_input = get_user_input_with_gasome()

# Configurar uma subhead e mostrar aos utilizadores input
st.subheader('User Input:')
st.write(user_input)

# Guardar o modelos preditos numa variavel
#prediction = clf.predict(user_input)

# configurar um subheader e mostrar a classificação
st.subheader('Classification:')
#st.write(prediction)






