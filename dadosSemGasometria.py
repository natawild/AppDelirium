import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


def get_dataset(name):
    if name == "Com Gasometria":
        return pd.read_csv("./DeliriumcomGasometria.csv")
    return pd.read_csv("./DeliriumsemGasometria.csv")


def convertCheckboxToInt(variavel):
    if variavel == 1:
        return 1
    return 0


# TODO: confirmar valores a baixo
def convertGenderToInt(variavel):
    if variavel == "Masculino":
        return 0
    return 1


def get_user_input_without_gasome():
    proveniencias = st.sidebar.slider("Proveniencia", 0, 5, 1)
    idades = st.sidebar.slider("Idade", 18, 120, 1)
    genders = st.sidebar.radio("Selecione o sexo:", ("Masculino", "Feminino"))
    tempos = st.sidebar.slider("Tempo em horas", 0, 15, 1)
    glicoses = st.sidebar.slider("glicose", 20, 1000, 1)
    sodios = st.sidebar.slider("sodio", 100, 170, 1)
    ureias = st.sidebar.slider("ureia", 1, 280, 1)
    creatininas = st.sidebar.slider(
        "creatinina", min_value=0.10, max_value=20.00, step=0.01
    )
    pcrs = st.sidebar.slider("pcr", min_value=2.90, max_value=500.00, step=0.01)
    rosuvastatinas = st.sidebar.checkbox("Rosuvastatina", help="WIP: Define")
    atorvastatinas = st.sidebar.checkbox("Atorvastatina")
    pravastatinas = st.sidebar.checkbox("Pravastatina")
    sinvastatinas = st.sidebar.checkbox("Sinvastatina")
    fluvastatinas = st.sidebar.checkbox("Fluvastatina")
    alprazolams = st.sidebar.checkbox("Alprazolam")
    captoprils = st.sidebar.checkbox("Captopril")
    codeines = st.sidebar.checkbox("Codeine")
    desloratadines = st.sidebar.checkbox("Desloratadine")
    diazepams = st.sidebar.checkbox("Diazepam", help="Unisedil, Valium")
    lorazepams = st.sidebar.checkbox("Lorazepam")
    digoxins = st.sidebar.checkbox("Digoxin")
    dipyridamoles = st.sidebar.checkbox("Dipyridamole")
    furosemides = st.sidebar.checkbox("Furosemide")
    fluvoxamines = st.sidebar.checkbox("Fluvoxamine")
    haloperidols = st.sidebar.checkbox("Haloperidol", help="Haldol")
    hydrocortisones = st.sidebar.checkbox("Hydrocortisone")
    iloperidones = st.sidebar.checkbox("Iloperidone")
    morphines = st.sidebar.checkbox("Morphine")
    nifedipines = st.sidebar.checkbox("Nifedipine")
    paliperidones = st.sidebar.checkbox("Paliperidone")
    prednisones = st.sidebar.checkbox("Prednisone")
    ranitidines = st.sidebar.checkbox("Ranitidine")
    risperidones = st.sidebar.checkbox("Risperidone")
    trazodones = st.sidebar.checkbox("Trazodone", help="Triticum")
    venlafaxines = st.sidebar.checkbox("Venlafaxine")
    warfarins = st.sidebar.checkbox("Warfarin")
    amitriptylines = st.sidebar.checkbox("Amitriptyline", help="Saroten")
    hydroxyzines = st.sidebar.checkbox("Hydroxyzine")
    paroxetines = st.sidebar.checkbox(
        "Paroxetine", help="Seroxat; Paxetil; Calmus; Denerval; Oxepar"
    )
    quetiapines = st.sidebar.checkbox("Quetiapine")
    scopolamines = st.sidebar.checkbox("Scopolamine", help="Buscopan")
    trihexyphenidyls = st.sidebar.checkbox("Trihexyphenidyl")
    clonidines = st.sidebar.checkbox("Clonidine")
    sertralinas = st.sidebar.checkbox("Sertralina")
    tramadols = st.sidebar.checkbox("Tramadol")
    mexazolams = st.sidebar.checkbox("Mexazolam", help="Sedoxil")
    trospiums = st.sidebar.checkbox("Trospium")
    alcoolicos = st.sidebar.slider("Alcoolico", 0, 1, 0)

    # Guardar o dicionário numa variável
    user_data = {
        "proveniencia": proveniencias,
        "idade": idades,
        "gender": convertGenderToInt(genders),
        "tempo": tempos,
        "glicose": glicoses,
        "sodio": sodios,
        "ureia": ureias,
        "creatinina": creatininas,
        "pcr": pcrs,
        "rosuvastatina": convertCheckboxToInt(rosuvastatinas),
        "atorvastatina": convertCheckboxToInt(atorvastatinas),
        "pravastatina": convertCheckboxToInt(pravastatinas),
        "sinvastatina": convertCheckboxToInt(sinvastatinas),
        "fluvastatina": convertCheckboxToInt(fluvastatinas),
        "alprazolam": convertCheckboxToInt(alprazolams),
        "captopril": convertCheckboxToInt(captoprils),
        "codeine": convertCheckboxToInt(codeines),
        "desloratadine": convertCheckboxToInt(desloratadines),
        "diazepam": convertCheckboxToInt(diazepams),
        "lorazepam": convertCheckboxToInt(lorazepams),
        "digoxin": convertCheckboxToInt(digoxins),
        "dipyridamole": convertCheckboxToInt(dipyridamoles),
        "furosemide": convertCheckboxToInt(furosemides),
        "fluvoxamine": convertCheckboxToInt(fluvoxamines),
        "haloperidol": convertCheckboxToInt(haloperidols),
        "hydrocortisone": convertCheckboxToInt(hydrocortisones),
        "iloperidone": convertCheckboxToInt(iloperidones),
        "morphine": convertCheckboxToInt(morphines),
        "nifedipine": convertCheckboxToInt(nifedipines),
        "paliperidone": convertCheckboxToInt(paliperidones),
        "prednisone": convertCheckboxToInt(prednisones),
        "ranitidine": convertCheckboxToInt(ranitidines),
        "risperidone": convertCheckboxToInt(risperidones),
        "trazodone": convertCheckboxToInt(trazodones),
        "venlafaxine": convertCheckboxToInt(venlafaxines),
        "warfarin": convertCheckboxToInt(warfarins),
        "amitriptyline": convertCheckboxToInt(amitriptylines),
        "hydroxyzine": convertCheckboxToInt(hydroxyzines),
        "paroxetine": convertCheckboxToInt(paroxetines),
        "quetiapine": convertCheckboxToInt(quetiapines),
        "scopolamine": convertCheckboxToInt(scopolamines),
        "trihexyphenidyl": convertCheckboxToInt(trihexyphenidyls),
        "clonidine": convertCheckboxToInt(clonidines),
        "sertralina": convertCheckboxToInt(sertralinas),
        "tramadol": convertCheckboxToInt(tramadols),
        "mexazolam": convertCheckboxToInt(mexazolams),
        "trospium": convertCheckboxToInt(trospiums),
        "alcoolico": alcoolicos,
    }

    # Transformar os dados num dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features
