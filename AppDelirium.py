import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dadosComGasometria import get_user_input_with_gasome
import joblib

#Page headers
st.set_page_config(
    page_title='Delirium Detection', 
    page_icon=None, 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None
)

#Page Title
st.title('Delirium Detection')


# load the model from disk
clf = joblib.load('final_model.sav')

#Criacao de um título e subtitulo
st.write("""
App Delirium
predict if someone has delirium using machine learning and python !
""")

filters_container = st.container()

filters_container.subheader("Formulario")
filters_container.write("Por favor preencha dos dados a baixo para poder efectuar uma previsão de delirium")
fcol1, fcol2, fcol3 = filters_container.columns(3)
#antidislipidemicos = fcol1.multiselect(
#    'Antidislipidemicos',
#    ['Rosuvastatina', 'Atorvastatina', 'Pravastatina', 'Sinvastatina', 'Fluvastatina'],
#    default=None,
#    help="HEP_TEXT"
#),

#fcol1.subheader("A wide column with a chart")
#fcol2.subheader("A wide column with a chart")
#fcol3.subheader("A wide column with a chart")

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

user_input = get_user_input_with_gasome(fcol1, fcol2, fcol3)

# Configurar uma subhead e mostrar aos utilizadores input
st.subheader('User Input:')
st.write(user_input)

# Guardar o modelos preditos numa variavel
#prediction = clf.predict(user_input)

# configurar um subheader e mostrar a classificação
st.subheader('Classification:')
#st.write(prediction)






