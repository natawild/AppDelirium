import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

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


def get_dataset(name):
    dadosComGasometria = None
    dadosSemGasometria = None
    if name == 'Com Gasometria':
        dadosComGasometria = pd.read_csv('./DeliriumcomGasometria.csv')
        X = dadosComGasometria.iloc[:, 0:52].values
        y = dadosComGasometria.iloc[:, -1].values
    else :
        dadosSemGasometria = pd.read_csv('./DeliriumsemGasometria.csv')
        X = dadosSemGasometria.iloc[:, 0:50].values
        y = dadosSemGasometria.iloc[:, -1].values
    return X, y



X, y = get_dataset(dataset_name)
# configurar um sub titulo
st.subheader('Data Information:')
st.write('Shape of dataset:', X.shape)
st.write('Head of dataset:', X) 
st.write('Head of dataset:', y) 
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression()
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
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


gender = get_dataset(dataset_name)
st.write(gender)

 

def get_user_input_with_gasome():
    proveniencia = st.sidebar.slider('Proveniencia', 0, 5, 1)
    idade = st.sidebar.slider('Idade', 18, 120, 1)
    gender = st.sidebar.slider('Selecione o sexo:', 0, 1, 1)
    tempo = st.sidebar.slider('Tempo em horas', 0, 15, 1)
    glicose = st.sidebar.slider('glicose', 20, 1000, 1)
    sodio = st.sidebar.slider('sodio', 100, 170, 1)
    ureia = st.sidebar.slider('ureia', 1, 280, 1)
    creatinina = st.sidebar.slider('creatinina', min_value=0.10, max_value=20.00, step=0.01)
    pcr = st.sidebar.slider('pcr', min_value=2.90, max_value=500.00, step=0.01)
    ph = st.sidebar.slider('ph',min_value=7.00, max_value=7.770, step=0.001)
    ca = st.sidebar.slider('ca', min_value=0.50, max_value=1.40, step=0.01)
    co2 = st.sidebar.slider('co2', min_value=10.00, max_value=130.00, step=0.01)
    o2 = st.sidebar.slider('o2', min_value=30.00, max_value=180.00, step=0.01)
    hco3 = st.sidebar.slider('hco3', min_value=3.00, max_value=48.00, step=0.01)
    rosuvastatina = st.sidebar.slider('Rosuvastatina', 0, 1, 0)
    atorvastatina = st.sidebar.slider('Atorvastatina', 0, 1, 0)
    pravastatina = st.sidebar.slider('Pravastatina', 0, 1, 0)
    sinvastatina = st.sidebar.slider('Sinvastatina', 0, 1, 1)
    fluvastatina = st.sidebar.slider('Fluvastatina', 0, 1, 1)
    alprazolam = st.sidebar.slider('Alprazolam', 0, 1, 1)
    captopril = st.sidebar.slider('Captopril', 0, 1, 1)
    codeine = st.sidebar.slider('Codeine', 0, 1, 1)
    desloratadine = st.sidebar.slider('Desloratadine', 0, 1, 1)
    diazepam = st.sidebar.slider('Diazepam', 0, 1, 1)
    lorazepam = st.sidebar.slider('Lorazepam', 0, 1, 1)
    digoxin = st.sidebar.slider('Digoxin', 0, 1, 1)
    dipyridamole = st.sidebar.slider('Dipyridamole', 0, 1, 1)
    furosemide = st.sidebar.slider('Furosemide', 0, 1, 1)
    fluvoxamine = st.sidebar.slider('Fluvoxamine', 0, 1, 1)
    haloperidol = st.sidebar.slider('Haloperidol', 0, 1, 1)
    hydrocortisone = st.sidebar.slider('Hydrocortisone', 0, 1, 1)
    iloperidone = st.sidebar.slider('Iloperidone', 0, 1, 1)
    morphine = st.sidebar.slider('Morphine', 0, 1, 1)
    nifedipine = st.sidebar.slider('Nifedipine', 0, 1, 1)
    paliperidone = st.sidebar.slider('Paliperidone', 0, 1, 1)
    prednisone = st.sidebar.slider('Prednisone', 0, 1, 1)
    ranitidine = st.sidebar.slider('Ranitidine', 0, 1, 1)
    risperidone = st.sidebar.slider('Risperidone', 0, 1, 1)
    trazodone = st.sidebar.slider('Trazodone', 0, 1, 1)
    venlafaxine = st.sidebar.slider('Venlafaxine', 0, 1, 1)
    warfarin = st.sidebar.slider('Warfarin', 0, 1, 1)
    amitriptyline = st.sidebar.slider('Amitriptyline', 0, 1, 1)
    hydroxyzine = st.sidebar.slider('Hydroxyzine', 0, 1, 1)
    paroxetine = st.sidebar.slider('Paroxetine', 0, 1, 1)
    quetiapine = st.sidebar.slider('Quetiapine', 0, 1, 1)
    scopolamine = st.sidebar.slider('Scopolamine', 0, 1, 1)
    trihexyphenidyl = st.sidebar.slider('Trihexyphenidyl', 0, 1, 1)
    clonidine = st.sidebar.slider('Clonidine', 0, 1, 1)
    sertralina = st.sidebar.slider('Sertralina', 0, 1, 1)
    tramadol = st.sidebar.slider('Tramadol', 0, 1, 1)
    mexazolam = st.sidebar.slider('Mexazolam', 0, 1, 1)
    trospium = st.sidebar.slider('Trospium', 0, 1, 1)
    alcoolico = st.sidebar.slider('Alcoolico', 0, 1, 1)



    # Guardar o dicionário numa variável
    user_data = {'proveniencia': proveniencia,
                'idade': idade,
                'gender': gender,
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
                'rosuvastatina' : rosuvastatina,
                'atorvastatina' : atorvastatina, 
                'pravastatina': pravastatina,
                'sinvastatina': sinvastatina,
                'fluvastatina' : fluvastatina, 
                'alprazolam' : alprazolam, 
                'captopril' : captopril, 
                'codeine' : codeine, 
                'desloratadine' : desloratadine, 
                'diazepam' : diazepam, 
                'lorazepam': lorazepam,
                'digoxin': digoxin, 
                'dipyridamole' : dipyridamole, 
                'furosemide' : furosemide, 
                'fluvoxamine' : fluvoxamine, 
                'haloperidol' : haloperidol, 
                'hydrocortisone' : hydrocortisone, 
                'iloperidone' : iloperidone, 
                'morphine' : morphine, 
                'nifedipine' : nifedipine, 
                'paliperidone' : paliperidone, 
                'prednisone' : prednisone, 
                'ranitidine' : ranitidine, 
                'risperidone' : risperidone, 
                'trazodone' : trazodone, 
                'venlafaxine' : venlafaxine, 
                'warfarin' : warfarin, 
                'amitriptyline' : amitriptyline, 
                'hydroxyzine' : hydroxyzine, 
                'paroxetine' : paroxetine, 
                'quetiapine' : quetiapine, 
                'scopolamine' : scopolamine, 
                'trihexyphenidyl' : trihexyphenidyl, 
                'clonidine' : clonidine, 
                'sertralina' : sertralina, 
                'tramadol' : tramadol, 
                'mexazolam' : mexazolam, 
                'trospium' : trospium,  
                'alcoolico': alcoolico

                 }

    # Transformar os dados num dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


# guardar o input do utilizador numa variavel

user_input = get_user_input_with_gasome()

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

