import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

def get_dataset(name):
    if name == 'Sem Gasometria':
        return pd.read_csv('./DeliriumsemGasometria.csv')
    return pd.read_csv('./DeliriumcomGasometria.csv')

def convertCheckboxToInt(variavel):
    if variavel == 1:
        return 1
    return 0

# TODO: confirmar valores a baixo
def convertGenderToInt(variavel):
    if variavel == 'Masculino':
        return 0
    return 1


def get_user_input_with_gasome():
    proveniencia = st.sidebar.slider('Proveniencia', 0, 5, 1)
    idade = st.sidebar.slider('Idade', 18, 120, 1)
    gender = st.sidebar.radio('Selecione o sexo:', ('Masculino', 'Feminino'))
    tempo = st.sidebar.slider('Tempo em horas', 0, 15, 1)
    glicose = st.sidebar.slider('glicose', 20, 1000, 1)
    sodio = st.sidebar.slider('sodio', 100, 170, 1)
    ureia = st.sidebar.slider('ureia', 1, 280, 1)
    creatinina = st.sidebar.slider('creatinina', min_value=0.10, max_value=20.00, step=0.01)
    pcr = st.sidebar.slider('pcr', min_value=2.90, max_value=500.00, step=0.01)
    ph = st.sidebar.slider('ph',min_value=7.011, max_value=7.779, step=0.001)
    ca = st.sidebar.slider('ca', min_value=0.50, max_value=1.40, step=0.01)
    co2 = st.sidebar.slider('co2', min_value=10.00, max_value=130.00, step=0.01)
    o2 = st.sidebar.slider('o2', min_value=30.00, max_value=180.00, step=0.01)
    hco3 = st.sidebar.slider('hco3', min_value=3.00, max_value=48.00, step=0.01)
    rosuvastatina = st.sidebar.checkbox('Rosuvastatina', help = 'WIP: Define')
    atorvastatina = st.sidebar.checkbox('Atorvastatina')
    pravastatina = st.sidebar.checkbox('Pravastatina')
    sinvastatina = st.sidebar.checkbox('Sinvastatina')
    fluvastatina = st.sidebar.checkbox('Fluvastatina')
    alprazolam = st.sidebar.checkbox('Alprazolam')
    captopril = st.sidebar.checkbox('Captopril')
    codeine = st.sidebar.checkbox('Codeine')
    desloratadine = st.sidebar.checkbox('Desloratadine')
    diazepam = st.sidebar.checkbox('Diazepam', help='Unisedil, Valium')
    lorazepam = st.sidebar.checkbox('Lorazepam')
    digoxin = st.sidebar.checkbox('Digoxin')
    dipyridamole = st.sidebar.checkbox('Dipyridamole')
    furosemide = st.sidebar.checkbox('Furosemide')
    fluvoxamine = st.sidebar.checkbox('Fluvoxamine')
    haloperidol = st.sidebar.checkbox('Haloperidol', help ='Haldol')
    hydrocortisone = st.sidebar.checkbox('Hydrocortisone')
    iloperidone = st.sidebar.checkbox('Iloperidone')
    morphine = st.sidebar.checkbox('Morphine')
    nifedipine = st.sidebar.checkbox('Nifedipine')
    paliperidone = st.sidebar.checkbox('Paliperidone')
    prednisone = st.sidebar.checkbox('Prednisone')
    ranitidine = st.sidebar.checkbox('Ranitidine')
    risperidone = st.sidebar.checkbox('Risperidone')
    trazodone = st.sidebar.checkbox('Trazodone', help='Triticum')
    venlafaxine = st.sidebar.checkbox('Venlafaxine')
    warfarin = st.sidebar.checkbox('Warfarin')
    amitriptyline = st.sidebar.checkbox('Amitriptyline')
    hydroxyzine = st.sidebar.checkbox('Hydroxyzine')
    paroxetine = st.sidebar.checkbox('Paroxetine', help='Seroxat, Paxetil, Calmus, Denerval e Oxepar')
    quetiapine = st.sidebar.checkbox('Quetiapine')
    scopolamine = st.sidebar.checkbox('Scopolamine', help='Buscopan')
    trihexyphenidyl = st.sidebar.checkbox('Trihexyphenidyl')
    clonidine = st.sidebar.checkbox('Clonidine')
    sertralina = st.sidebar.checkbox('Sertralina')
    tramadol = st.sidebar.checkbox('Tramadol')
    mexazolam = st.sidebar.checkbox('Mexazolam', help= 'Sedoxil')
    trospium = st.sidebar.checkbox('Trospium')
    alcoolico = st.sidebar.slider('Alcoolico', 0, 1, 1)

    # Guardar o dicionário numa variável
    user_data = {'proveniencia': proveniencia,
                'idade': idade,
                'gender': convertGenderToInt(gender),
                'tempo': tempo,
                'glicose': glicose,
                'sodio': sodio,
                'ureia': ureia,
                'creatinina': creatinina,
                'pcr' : pcr, 
                'ph' : ph,
                'ca' : ca,
                'ureia' : ureia,
                'co2': co2,
                'hco3': hco3,
                'rosuvastatina' : convertCheckboxToInt(rosuvastatina),
                'atorvastatina' : convertCheckboxToInt(atorvastatina), 
                'pravastatina': convertCheckboxToInt(pravastatina),
                'sinvastatina': convertCheckboxToInt(sinvastatina),
                'fluvastatina' : convertCheckboxToInt(fluvastatina), 
                'alprazolam' : convertCheckboxToInt(alprazolam), 
                'captopril' : convertCheckboxToInt(captopril), 
                'codeine' : convertCheckboxToInt(codeine), 
                'desloratadine' : convertCheckboxToInt(desloratadine), 
                'diazepam' : convertCheckboxToInt(diazepam), 
                'lorazepam': convertCheckboxToInt(lorazepam),
                'digoxin': convertCheckboxToInt(digoxin), 
                'dipyridamole' : convertCheckboxToInt(dipyridamole), 
                'furosemide' : convertCheckboxToInt(furosemide), 
                'fluvoxamine' : convertCheckboxToInt(fluvoxamine), 
                'haloperidol' : convertCheckboxToInt(haloperidol), 
                'hydrocortisone' : convertCheckboxToInt(hydrocortisone), 
                'iloperidone' : convertCheckboxToInt(iloperidone), 
                'morphine' : convertCheckboxToInt(morphine), 
                'nifedipine' : convertCheckboxToInt(nifedipine), 
                'paliperidone' : convertCheckboxToInt(paliperidone), 
                'prednisone' : convertCheckboxToInt(prednisone), 
                'ranitidine' : convertCheckboxToInt(ranitidine), 
                'risperidone' : convertCheckboxToInt(risperidone), 
                'trazodone' : convertCheckboxToInt(trazodone), 
                'venlafaxine' : convertCheckboxToInt(venlafaxine), 
                'warfarin' : convertCheckboxToInt(warfarin), 
                'amitriptyline' : convertCheckboxToInt(amitriptyline), 
                'hydroxyzine' : convertCheckboxToInt(hydroxyzine), 
                'paroxetine' : convertCheckboxToInt(paroxetine), 
                'quetiapine' : convertCheckboxToInt(quetiapine), 
                'scopolamine' : convertCheckboxToInt(scopolamine), 
                'trihexyphenidyl' : convertCheckboxToInt(trihexyphenidyl), 
                'clonidine' : convertCheckboxToInt(clonidine), 
                'sertralina' : convertCheckboxToInt(sertralina), 
                'tramadol' : convertCheckboxToInt(tramadol), 
                'mexazolam' : convertCheckboxToInt(mexazolam), 
                'trospium' : convertCheckboxToInt(trospium),  
                'alcoolico': alcoolico

                    }

    # Transformar os dados num dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

    