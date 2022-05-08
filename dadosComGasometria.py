import streamlit as st
import numpy as np
import pandas as pd


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



def get_user_input_with_gasome(fcol1, fcol2, fcol3, fcol4):
    
    proveniencia = fcol1.selectbox("Local de Proveniencia", ("Casa", "Inter-Hospitalar", "Intra-Hospitalar", "Lar"))
    grupoDiagnostico = fcol1.selectbox("Grupo de Diagnóstico", ("Hemato-Oncologico","Neurologico","Respiratorio","Cardiovascular","Musculo-esqueletico","Geniturinário","Gastrointestinal","Outro"))
    localSU = fcol1.selectbox("Tipo de admissão",("Ambulatório","UCISU","UDC1","UDC2"))
    idade = fcol1.slider("Idade", min_value=18, max_value=100, step=1)
    gender = fcol1.radio("Selecione o sexo:", ("Masculino", "Feminino"))
    tempo = fcol1.number_input("Tempo de permanência no SU", min_value=0.08, max_value=12.0, step=0.01 ,help="Em dias")
    sirs = fcol2.slider("Critérios SIRS:",min_value=0, max_value=4, step=1, help="Temperatura corporal, Frequência respiratória, Frequência cardíaca, Número de leucócitos")
    glicose = fcol2.number_input("Glicose (mg/dL)", min_value=41.0, max_value=1000.0, step=0.01, value=90.0)
    sodio = fcol2.number_input("Sódio (mEq/L)", min_value=42.0, max_value=151.0, step=0.01, value= 136.0)
    ureia = fcol2.number_input("Ureia (mg/dL)", min_value=4.0, max_value=275.0, step=0.01, value = 21.0)
    creatinina = fcol2.number_input(
        "Creatinina (mg/dL)", min_value=0.1, max_value=19.5, step=0.01, value=0.8
    )
    pcr = fcol2.number_input("PCR (mg/L)", min_value=2.90, max_value=499.00, step=0.01)
    ph = fcol2.number_input("pH", min_value=7.026, max_value=7.625, step=0.001, value=7.38)
    ca = fcol3.number_input("Cálcio ionizado (mmol/L)", min_value=0.84, max_value=1.37, step=0.01, value=1.21)
    co2 = fcol3.number_input("Pressão parcial de dióxido de carbono (mm Hg)", min_value=13.2, max_value=121.3, step=0.01, value=36.3)
    o2 = fcol3.number_input("Pressão parcial de oxigénio (mm Hg)", min_value=34.1, max_value=178.1, step=0.01, value=87.9)
    hco3 = fcol3.number_input("Ião bicarbonato (mEq/L)", min_value=7.40, max_value=39.1, step=0.01, value=24.6)

    antidislipidemicos = fcol3.multiselect(
        'Antidislipidemicos',
        ['Rosuvastatina', 'Atorvastatina', 'Pravastatina', 'Sinvastatina', 'Fluvastatina'],
        default=None,
        help="Rosuvastatina, Atorvastatina, Pravastatina, Sinvastatina, Fluvastatina"
    ),
    antipsicoticos = fcol3.multiselect(
        'Antipsicóticos',
        ['Haloperidol', 'Quetiapine', 'Risperidone', 'Paliperidone', 'Iloperidone'],
        default=None,
        help="Haloperidol, Quetiapine, Risperidone, Paliperidone, Iloperidone"
    ),
    antidepressores = fcol4.multiselect(
        'Antidepressores',
        ['Fluvoxamine','Paroxetine', 'Sertralina', 'Venlafaxine', 'Trazodone', 'Amitriptyline'],
        default=None,
        help="Fluvoxamine, Paroxetine, Sertralina, Venlafaxine, Trazodone, Amitriptyline"
    ),
    #antihipertensores = fcol3.multiselect(
    #    'Antihipertensores',
    #    ['Nifedipine','Captopril','Clonidine'],
    #    default=None,
    #    help="HEP_TEXT"
    #),
    analgesicos = fcol4.multiselect(
        'Analgésicos',
        ['Nifedipine','Captopril','Clonidine'],
        default=None,
        help="Nifedipine, Captopril, Clonidine"
    ),
    anticoagulantes = fcol4.multiselect(
        'Anticoagulantes',
        ['Warfarin','Dipyridamole'],
        default=None,
        help="Warfarin, Dipyridamole"
    ),
    corticosteroides = fcol4.multiselect(
        'Corticosteroides',
        ['Hydrocortisone','Prednisone'],
        default=None,
        help="Hydrocortisone, Prednisone"
    ),
    digitalicos = fcol4.multiselect(
        'Digitálicos',
        ['Digoxin'],
        default=None,
        help="Digoxin"
    ),
    outrosMed = fcol4.multiselect(
        'Outros Medicamentos',
        ['Ranitidine','Scopolamine', 'Desloratadine', 'Hydroxyzine', 'Trihexyphenidyl', 'Trospium'],
        default= None,
        help="Ranitidine, Scopolamine, Desloratadine, Hydroxyzine, Trihexyphenidyl, Trospium"
    ),
    
    alcoolico = fcol1.radio("Consumo de alcool em excesso?", ["Sim", "Não"], index=1)

    # Guardar o dicionário numa variável
    user_data = {
        "Proveniencia": proveniencia,
        "Grupo de Diagnostico": grupoDiagnostico,
        "Tipo de admissão": localSU,
        "Idade": idade,
        "Género": gender,
        "Tempo": tempo,
        "Critérios SIRS" : sirs,
        "Glicose": glicose,
        "Sódio": sodio,
        "Ureia": ureia,
        "Creatinina": creatinina,
        "PCR": pcr,
        "pH": ph,
        "ca": ca,
        "CO2": co2,
        "O2": o2,
        "HCO3": hco3,
        "Antidislipidemicos":antidislipidemicos,
        "Antipsicoticos":antipsicoticos,
        "Antidepressores":antidepressores,
        "Analgésicos":analgesicos,
        "Anticoagulantes": anticoagulantes,
        "Antidepressores": antidepressores,
        "Corticosteroides":corticosteroides,
        "Digitalicos":digitalicos,
        "Outros Medicamentos":outrosMed, 
        "Consumo de álcool": alcoolico,  
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
        "Idade": normalize(features["Idade"],18,100),
        "Género": convertGenderToInt(features["Género"]),
        "Interna_Dias": normalize(features["Tempo"],0.083,12),
        "SIRS" : normalize(features["Critérios SIRS"],0,4),
        "Glicose": normalize(features["Glicose"],41,1000),
        "Sodio": normalize(features["Sódio"],42,151),
        "Ureia": normalize(features["Ureia"],4,275),
        "Creatinina": normalize(features["Creatinina"],0.1,19.5),
        "PCR": normalize(features["PCR"],2.3,499),
        "pH": normalize(features["pH"],7.026,7.625),
        "Ca_ionizado": normalize(features["ca"],0.84,1.37),
        "pCO2": normalize(features["CO2"],13.2,121.3),
        "pO2": normalize(features["O2"],34.1,178.1),
        "HCO3": normalize(features["HCO3"],7.40,39.1),
        "Tipo de admissão": convertLocalSu(features["Tipo de admissão"]),
        "Antidislipidemicos": convertMultiSelect(features["Antidislipidemicos"]),
        "Antipsicoticos": convertMultiSelect(features["Antipsicoticos"]),
        "Antidepressores": convertMultiSelect(features["Antidepressores"]),
        "Analgesicos": convertMultiSelect(features["Analgésicos"]),
        "Anticoagulantes": convertMultiSelect(features["Anticoagulantes"]),
        "Alcoolico": convertMultiSelect(features["Consumo de álcool"]),
        "Corticosteroides": convertMultiSelect(features["Corticosteroides"]),
        "Digitalicos": convertMultiSelect(features["Digitalicos"]),
        "Outros Med_Presente": convertMultiSelect(features["Outros Medicamentos"]),
    }

    merged = {** data_to_predict, **convertProv(features["Proveniencia"])}
    merged = {** merged, **convertGrupoDiag(features["Grupo de Diagnostico"])}

    return pd.DataFrame(merged, index=[0])

