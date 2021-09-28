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
    rosuvastatina = st.sidebar.checkbox('Rosuvastatina')
    atorvastatina = st.sidebar.checkbox('Atorvastatina', value = 1 )
    pravastatina = st.sidebar.checkbox('Pravastatina')
    sinvastatina = st.sidebar.checkbox('Sinvastatina')
    fluvastatina = st.sidebar.checkbox('Fluvastatina')
    alprazolam = st.sidebar.checkbox('Alprazolam')
    captopril = st.sidebar.checkbox('Captopril')
    codeine = st.sidebar.checkbox('Codeine')
    desloratadine = st.sidebar.checkbox('Desloratadine')
    diazepam = st.sidebar.checkbox('Diazepam')
    lorazepam = st.sidebar.checkbox('Lorazepam')
    digoxin = st.sidebar.checkbox('Digoxin')
    dipyridamole = st.sidebar.checkbox('Dipyridamole')
    furosemide = st.sidebar.checkbox('Furosemide')
    fluvoxamine = st.sidebar.checkbox('Fluvoxamine')
    haloperidol = st.sidebar.checkbox('Haloperidol')
    hydrocortisone = st.sidebar.checkbox('Hydrocortisone')
    iloperidone = st.sidebar.checkbox('Iloperidone')
    morphine = st.sidebar.checkbox('Morphine')
    nifedipine = st.sidebar.checkbox('Nifedipine')
    paliperidone = st.sidebar.checkbox('Paliperidone')
    prednisone = st.sidebar.checkbox('Prednisone')
    ranitidine = st.sidebar.checkbox('Ranitidine')
    risperidone = st.sidebar.checkbox('Risperidone')
    trazodone = st.sidebar.checkbox('Trazodone')
    venlafaxine = st.sidebar.checkbox('Venlafaxine')
    warfarin = st.sidebar.checkbox('Warfarin')
    amitriptyline = st.sidebar.checkbox('Amitriptyline')
    hydroxyzine = st.sidebar.checkbox('Hydroxyzine')
    paroxetine = st.sidebar.checkbox('Paroxetine')
    quetiapine = st.sidebar.checkbox('Quetiapine')
    scopolamine = st.sidebar.checkbox('Scopolamine')
    trihexyphenidyl = st.sidebar.checkbox('Trihexyphenidyl')
    clonidine = st.sidebar.checkbox('Clonidine')
    sertralina = st.sidebar.checkbox('Sertralina')
    tramadol = st.sidebar.checkbox('Tramadol')
    mexazolam = st.sidebar.checkbox('Mexazolam')
    trospium = st.sidebar.checkbox('Trospium')
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

