import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dadosComGasometria import get_user_input_with_gasome, convert_user_input_data_to_predict_format
import joblib

#Page headers
st.set_page_config(
    page_title='Delirium Detection', 
    page_icon=None, 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': None
     }
)

#remover side menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

#Page Title
st.title('Delirium Detection')

#change_text = """
#    <style>
#    div.stMultiSelect .st-ek {visibility: hidden;}
#    div.stMultiSelect .st-ek:before {content: "Selecione uma opção"; visibility: visible;}
#    </style>

#st.markdown(change_text, unsafe_allow_html=True)


header_container = st.container()
filters_container = st.container()
results_container = st.container()


# load the model from disk
clf = joblib.load('final_model.sav')

#Criacao de um título e subtitulo
header_container.write("""
App Delirium
predict if someone has delirium using machine learning and python !
""")

filters_container.subheader("Formulario")
filters_container.write("Por favor preencha dos dados a baixo para poder efectuar uma previsão de delirium")
fcol1, fcol2, fcol3 = filters_container.columns(3)

# guardar o input do utilizador numa variavel
user_input = get_user_input_with_gasome(fcol1, fcol2, fcol3)

# Configurar uma subhead e mostrar aos utilizadores input
results_container.subheader('Dados Introduzidos:')
results_container.write(user_input)

# Guardar o modelospd.DataFrame(data_to_predict, index=[0]) preditos numa variavel

# Exemplo de dados de entrada para o algoritmo de previsão 
# data = {'Casa': [0], 'Inter-Hospitalar ': [0], 'Intra-Hospitalar': [0], 'Lar': [0], 'GrupoDiagn_Cardiovascular': [0], 'GrupoDiagn_Gastrointestinal': [1], 'GrupoDiagn_Geniturinario': [0], 'GrupoDiagn_Hemato-Oncologico': [0], 'GrupoDiagn_Musculoesqueletico': [0], 'GrupoDiagn_Neurologico ': [0], 'GrupoDiagn_Outro': [0], 'GrupoDiagn_Respiratorio': [0], 'Local_SU': [2], 'Idade': [0.792682927], 'Interna_Dias': [0.034992028], 'SIRS': [0.0], 'Glicose': [0.057351408], 'Sodio': [0.798165138], 'Ureia': [0.712177122], 'Creatinina': [0.201030928], 'PCR': [0.128447755], 'pH': [0.607679466], 'Ca_ionizado': [0.339622642], 'pCO2': [0.301572618], 'pO2': [0.388194444], 'HCO3 ': [0.460567823], 'Genero': [1], 'Antidislipidemicos': [0], 'Antipsicoticos': [0], 'Antidepressores': [0], 'Analgesicos': [0], 'Anticoagulantes': [1], 'Digitalicos': [0], 'Corticosteroides': [0],'Outros Med_Presente': [0], 'Alcoolico': [0]}


def res(prediction):
    if prediction == 0:
        pred = 'Sem delirium'
    else:
        pred = 'Com delirum'
    return pred


def predictP():
    input_data_converted = convert_user_input_data_to_predict_format(user_input)
    prediction = clf.predict(input_data_converted)
    st.write('Previsão: ', res(prediction[0]))

results_container.button(
    label='Calcular Previsão',
    on_click=predictP()
)


#prediction = clf.predict(data_to_predict)
# configurar um subheader e mostrar a classificação
#results_container.subheader('Classification:')
#st.write(res(prediction))



