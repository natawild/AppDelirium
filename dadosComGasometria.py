import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


def convertCheckboxToInt(variavel):
    if variavel == 1:
        return 1
    return 0


def convertMultiSelect(values):
    if values.any() == False:
        return 0
    return 1

def convertGenderToInt(variavel):
    if variavel[0] == "Masculino":
        return 1
    return 0


def convertAlcool(variavel):
    if variavel[0] == "Sim":
        return 1
    return 0


def normalize(value, min, max):
    normalized = (value[0] - min) / (max - min)
    return normalized

    
#value = switcher.get(proveniencia, 0)



def get_user_input_with_gasome(fcol1, fcol2, fcol3):
    
    proveniencia = fcol1.selectbox("Local de Proveniencia", ("Casa", "Inter-Hospitalar", "Intra-Hospitalar", "Lar"))
    grupoDiagnostico = fcol1.selectbox("Grupo de Diagnóstico", ("Hemato-Oncologico","Neurologico","Respiratorio","Cardiovascular","Musculo-esqueletico","Geniturinário","Gastrointestinal","Outro"))
    localSU = fcol1.selectbox("Estado do doente",("Ambulatório","UCISU","UDC1","UDC2"))
    idade = fcol1.slider("Idade", min_value=18, max_value=100, step=1)
    gender = fcol1.radio("Selecione o sexo:", ("Masculino", "Feminino"))
    tempo = fcol1.number_input("Tempo de permanência no SU", min_value=0.08, max_value=12.0, step=0.01 ,help="Em dias")
    sirs = fcol1.slider("Critérios SIRS:",min_value=0, max_value=4, step=1, help="Temperatura corporal, Frequência respiratória, Frequência cardíaca, Número de leucócitos")
    glicose = fcol3.number_input("Glicose", min_value=41.0, max_value=1000.0, step=0.01)
    sodio = fcol2.number_input("Sódio", min_value=42.0, max_value=151.0, step=0.01)
    ureia = fcol2.number_input("Ureia", min_value=4.0, max_value=275.0, step=0.01)
    creatinina = fcol2.number_input(
        "Creatinina", min_value=0.1, max_value=19.5, step=0.01
    )
    pcr = fcol2.number_input("PCR", min_value=2.90, max_value=499.00, step=0.01)
    ph = fcol2.number_input("pH", min_value=7.026, max_value=7.625, step=0.001)
    ca = fcol2.number_input("Cálcio ionizado", min_value=0.84, max_value=1.37, step=0.01)
    co2 = fcol2.number_input("CO2", min_value=13.2, max_value=121.3, step=0.01)
    o2 = fcol2.number_input("O2", min_value=34.1, max_value=178.1, step=0.01)
    hco3 = fcol2.number_input("HCO3", min_value=7.40, max_value=39.1, step=0.01)

    antidislipidemicos = fcol3.multiselect(
        'Antidislipidemicos',
        ['Rosuvastatina', 'Atorvastatina', 'Pravastatina', 'Sinvastatina', 'Fluvastatina'],
        default=None,
    ),
    antipsicoticos = fcol3.multiselect(
        'Antipsicóticos',
        ['Haloperidol', 'Quetiapine', 'Risperidone', 'Paliperidone', 'Iloperidone'],
        default=None,
    ),
    antidepressores = fcol3.multiselect(
        'Antidepressores',
        ['Fluvoxamine','Paroxetine', 'Sertralina', 'Venlafaxine', 'Trazodone', 'Amitriptyline'],
        default=None,
    ),
    #antihipertensores = fcol3.multiselect(
    #    'Antihipertensores',
    #    ['Nifedipine','Captopril','Clonidine'],
    #    default=None,
    #    help="HEP_TEXT"
    #),
    analgesicos = fcol3.multiselect(
        'Analgésicos',
        ['Nifedipine','Captopril','Clonidine'],
        default=None,
        help="HEP_TEXT"
    ),
    anticoagulantes = fcol3.multiselect(
        'Anticoagulantes',
        ['Warfarin','Dipyridamole'],
        default=None,
        help="HEP_TEXT"
    ),
    corticosteroides = fcol3.multiselect(
        'Corticosteroides',
        ['Hydrocortisone','Prednisone'],
        default=None,
        help="HEP_TEXT"
    ),
    digitalicos = fcol3.multiselect(
        'Digitálicos',
        ['Digoxin'],
        default=None,
        help="HEP_TEXT"
    ),
    outrosMed = fcol3.multiselect(
        'Outros Medicamentos',
        ['Ranitidine','Scopolamine', 'Desloratadine', 'Hydroxyzine', 'Trihexyphenidyl', 'Trospium'],
        default= None,
        help="Ranitidine, Scopolamine, Desloratadine, Hydroxyzine, Trihexyphenidyl, Trospium'"
    ),
    
    alcoolico = fcol3.radio("Consumo de alcool em excesso?", ["Sim", "Não"])

    # Guardar o dicionário numa variável
    user_data = {
        "proveniencia": proveniencia,
        "grupoDiagnostico": grupoDiagnostico,
        "localSU": localSU,
        "idade": idade,
        "gender": gender,
        "tempo": tempo,
        "sirs" : sirs,
        "glicose": glicose,
        "sodio": sodio,
        "ureia": ureia,
        "creatinina": creatinina,
        "pcr": pcr,
        "ph": ph,
        "ca": ca,
        "co2": co2,
        "o2": o2,
        "hco3": hco3,
        "antidislipidemicos":antidislipidemicos,
        "antipsicoticos":antipsicoticos,
        "antidepressores":antidepressores,
        "analgesicos":analgesicos,
        "anticoagulantes": anticoagulantes,
        "antidepressores": antidepressores,
        "corticosteroides":corticosteroides,
        "digitalicos":digitalicos,
        "outrosMed":outrosMed, 
        "alcoolico": alcoolico,  
    }

    # Transformar os dados inseridos pelo utilizador num dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


def convertLocalSu(variavel):
    switcher = {
        'Ambulatório': 0,
        'UCISU': 1,
        'UDC1': 2,
        'UDC2': 3,
    }
    return switcher[variavel[0]]

def convertProv(variavel):
    dic = {
        'Casa': 1 if variavel[0] == 'Casa' else 0, 
        'Inter-Hospitalar': 1 if variavel[0] == 'Inter-Hospitalar' else 0,
        'Intra-Hospitalar': 1 if variavel[0] == 'Intra-Hospitalar' else 0,
        'Lar': 1 if variavel[0] == 'Lar' else 0,
    }
    return dic


def convertGrupoDiag(variavel):
    dic = {
        'GrupoDiagn_Hemato-Oncologico': 1 if variavel[0] == 'Hemato-Oncologico' else 0, 
        'GrupoDiagn_Neurologico': 1 if variavel[0] == 'Neurologico' else 0,
        'GrupoDiagn_Respiratorio': 1 if variavel[0] == 'Respiratorio' else 0,
        'GrupoDiagn_Musculoesqueletico': 1 if variavel[0] == 'Musculo-esqueletico' else 0,
        'GrupoDiagn_Cardiovascular': 1 if variavel[0] == 'Cardiovascular' else 0,
        'GrupoDiagn_Geniturinario': 1 if variavel[0] == 'Geniturinário' else 0,
        'GrupoDiagn_Gastrointestinal': 1 if variavel[0] == 'Gastrointestinal' else 0,
        'GrupoDiagn_Outro': 1 if variavel[0] == 'Outro' else 0,
    }
    return dic


def convert_user_input_data_to_predict_format(features):
# Guardar o dicionário numa variável
    data_to_predict = {
        "Idade": normalize(features["idade"],18,100),
        "Genero": convertGenderToInt(features["gender"]),
        "Interna_Dias": normalize(features["tempo"],0.083,12),
        "SIRS" : normalize(features["sirs"],0,4),
        "Glicose": normalize(features["glicose"],41,1000),
        "Sodio": normalize(features["sodio"],42,151),
        "Ureia": normalize(features["ureia"],4,275),
        "Creatinina": normalize(features["creatinina"],0.1,19.5),
        "PCR": normalize(features["pcr"],2.3,499),
        "pH": normalize(features["ph"],7.026,7.625),
        "Ca_ionizado": normalize(features["ca"],0.84,1.37),
        "pCO2": normalize(features["co2"],13.2,121.3),
        "pO2": normalize(features["o2"],34.1,178.1),
        "HCO3": normalize(features["hco3"],7.40,39.1),
        "Local_SU": convertLocalSu(features["localSU"]),
        "Antidislipidemicos": convertMultiSelect(features["antidislipidemicos"]),
        "Antipsicoticos": convertMultiSelect(features["antipsicoticos"]),
        "Antidepressores": convertMultiSelect(features["antidepressores"]),
        "Analgesicos": convertMultiSelect(features["analgesicos"]),
        "Anticoagulantes": convertMultiSelect(features["anticoagulantes"]),
        "Alcoolico": convertMultiSelect(features["alcoolico"]),
        "Corticosteroides": convertMultiSelect(features["corticosteroides"]),
        "Digitalicos": convertMultiSelect(features["digitalicos"]),
        "Outros Med_Presente": convertMultiSelect(features["outrosMed"]),
    }

    merged = {** data_to_predict, **convertProv(features["proveniencia"])}
    merged = {** merged, **convertGrupoDiag(features["grupoDiagnostico"])}

    return pd.DataFrame(merged, index=[0])
    '''
    # Transformar dados a prever num dataframe
    features_to_predict = pd.DataFrame(data_to_predict, index=[0])
    return features_to_predict
    '''


'''
# defining the function which will make the prediction using the data which the user inputs 
def prediction(proveniencia, idade, grupoDiagnostico, localSU, gender, tempo, sirs, glicose, sodio,ureia, creatinina, ph, ca, pcr, 
    co2, o2, hco3, antidislipidemicos, antipsicoticos, antidepressores, antihipertensores, analgesicos, 
    anticoagulantes,corticosteroides, digitalicos,outrosMed, alcoolico): 
    Casa == []
    InterHospitalar == []
    IntraHospitalar == []
    Lar == []
    GrupoDiagnCardiovascular == []
    GrupoDiagnGastrointestinal == []
    GrupoDiagnGeniturinario == []
    GrupoDiagn_Hemato_Oncologico== []
    GrupoDiagn_Musculoesqueletico== []
    GrupoDiagn_Neurologico== []
    GrupoDiagn_Outro == []
    GrupoDiagn_Respiratorio == []
    Local_SU = localSU, 
    Idade=normalize(idade,18,100)
    Interna_Dias=tempo
    SIRS=sirs
    Glicose =glicose
    Sodio=sodio
    Ureia=ureia
    Creatinina=creatinina
    PCR=pcr
    pH=ph
    Ca_ionizado=ca
    pCO2=co2
    pO2=o2
    HCO3=hco3
    Genero == []
    Antidislipidemicos == [],
    Antipsicoticos == []
    Antidepressores == []
    Analgesicos == []
    Anticoagulantes == []
    Digitalicos == []
    Corticosteroides == []
    Alcoolico == alcoolico
    OutrosMed_Presente == []
    
''' 
'''
variaveis que tenho de ter para a previsão: 
Casa, Inter-Hospitalar, Intra-Hospitalar, Lar, GrupoDiagn_Cardiovascular, GrupoDiagn_Gastrointestinal,
 GrupoDiagn_Geniturinario, GrupoDiagn_Hemato-Oncologico, GrupoDiagn_Musculoesqueletico, GrupoDiagn_Neurologico, GrupoDiagn_Outro,
  GrupoDiagn_Respiratorio, Local_SU, Idade, Interna_Dias, SIRS, Glicose, Sodio, Ureia, Creatinina, PCR, pH, Ca_ionizado, pCO2, 
  pO2, HCO3, Genero, Antidislipidemicos, Antipsicoticos, Antidepressores, Analgesicos, Anticoagulantes, Digitalicos, Corticosteroides, Outros Med_Presente, Alcoolico

'''
'''
    # Pre-processing user input    
if proveniencia == "Casa":
    Casa = 0
elif proveniencia == "Inter-Hospitalar":
    InterHospitalar = 1
elif proveniencia == "Intra-Hospitalar":
    IntraHospitalar = 2
else:
    Lar = 3


"Hemato-Oncologico","Neurologico","Respiratorio","Cardiovascular","Musculo-esqueletico","Geniturinário","Gastrointestinal","Outro"
if grupoDiagnostico == "Cardiovascular":
    grupodiagn = 0
elif grupoDiagnostico == "Gastrointestinal":
    GrupoDiagnCardiovascular = 1
elif grupoDiagnostico == "Geniturinário":
    IntraHospitalar = 2
elif grupoDiagnostico == "Hemato-Oncologico":
    GrupoDiagn_Hemato_Oncologico = 3
elif grupoDiagnostico == "Musculo-esqueletico":
    GrupoDiagn_Musculoesqueletico = 4
elif grupoDiagnostico == "Neurologico":
    GrupoDiagn_Neurologico = 5
elif grupoDiagnostico == "Respiratorio":
    GrupoDiagn_Respiratorio = 6 
else:
    GrupoDiagn_Outro = 7
 
# Pre-processing user input    
if gender == "Masculino":
    Genero = 0
else:
    Genero = 1
 

return Casa, InterHospitalar,IntraHospitalar, Lar, GrupoDiagnCardiovascular, GrupoDiagnGastrointestinal, GrupoDiagnGeniturinario, GrupoDiagn_Hemato_Oncologico, GrupoDiagn_Musculoesqueletico, GrupoDiagn_Neurologico, GrupoDiagn_Outro, GrupoDiagn_Respiratorio, Local_SU, Idade, Interna_Dias, SIRS, Glicose, Sodio, Ureia,Creatinina, PCR, pH, Ca_ionizado, pCO2, pO2, HCO3, Genero, Antidislipidemicos, Antipsicoticos, Antidepressores, Analgesicos, Anticoagulantes, Digitalicos, Corticosteroides, Outros Med_Presente, Alcoolico


    # Making predictions 
prediction = classifier.predict( 
    [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
if prediction == 0:
    pred = 'Rejected'
else:
    pred = 'Approved'
return pred


'''



