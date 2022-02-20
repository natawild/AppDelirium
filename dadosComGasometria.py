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


# TODO: confirmar valores a baixo
def convertGenderToInt(variavel):
    if variavel == "Masculino":
        return 0
    return 1



def get_user_input_with_gasome(fcol1, fcol2, fcol3):
    proveniencia = fcol1.slider("Proveniencia", 0, 5, 1)
    idade = fcol2.slider("Idade", 18, 120, 1)
    gender = fcol3.radio("Selecione o sexo:", ("Masculino", "Feminino"))
    tempo = fcol1.slider("Tempo em horas", 0, 15, 1)
    glicose = fcol2.slider("glicose", 20, 1000, 1)
    sodio = fcol3.slider("sodio", 100, 170, 1)
    ureia = fcol1.slider("ureia", 1, 280, 1)
    creatinina = fcol2.slider(
        "creatinina", min_value=0.10, max_value=20.00, step=0.01
    )
    pcr = fcol3.slider("pcr", min_value=2.90, max_value=500.00, step=0.01)
    ph = fcol1.slider("ph", min_value=7.011, max_value=7.779, step=0.001)
    ca = fcol2.slider("ca", min_value=0.50, max_value=1.40, step=0.01)
    co2 = fcol3.slider("co2", min_value=10.00, max_value=130.00, step=0.01)
    o2 = fcol1.slider("o2", min_value=30.00, max_value=180.00, step=0.01)
    hco3 = fcol2.slider("hco3", min_value=3.00, max_value=48.00, step=0.01)
    antidislipidemicos = fcol3.multiselect(
        'Antidislipidemicos',
        ['Rosuvastatina', 'Atorvastatina', 'Pravastatina', 'Sinvastatina', 'Fluvastatina'],
        default=None,
        help="HEP_TEXT"
    ),
    alprazolam = fcol1.checkbox("Alprazolam")
    captopril = fcol2.checkbox("Captopril")
    codeine = fcol3.checkbox("Codeine")
    desloratadine = fcol1.checkbox("Desloratadine")
    diazepam = fcol2.checkbox("Diazepam", help="Unisedil, Valium")
    lorazepam = fcol3.checkbox("Lorazepam")
    digoxin = fcol1.checkbox("Digoxin")
    dipyridamole = fcol2.checkbox("Dipyridamole")
    furosemide = fcol3.checkbox("Furosemide")
    fluvoxamine = fcol1.checkbox("Fluvoxamine")
    haloperidol = fcol2.checkbox("Haloperidol", help="Haldol")
    hydrocortisone = fcol3.checkbox("Hydrocortisone")
    iloperidone = fcol1.checkbox("Iloperidone")
    morphine = fcol2.checkbox("Morphine")
    nifedipine = fcol3.checkbox("Nifedipine")
    paliperidone = fcol1.checkbox("Paliperidone")
    prednisone = fcol2.checkbox("Prednisone")
    ranitidine = fcol3.checkbox("Ranitidine")
    risperidone = fcol1.checkbox("Risperidone")
    trazodone = fcol2.checkbox("Trazodone", help="Triticum")
    venlafaxine = fcol3.checkbox("Venlafaxine")
    warfarin = fcol1.checkbox("Warfarin")
    amitriptyline = fcol2.checkbox("Amitriptyline")
    hydroxyzine = fcol3.checkbox("Hydroxyzine")
    paroxetine = fcol1.checkbox(
        "Paroxetine", help="Seroxat, Paxetil, Calmus, Denerval e Oxepar"
    )
    quetiapine = fcol2.checkbox("Quetiapine")
    scopolamine = fcol3.checkbox("Scopolamine", help="Buscopan")
    trihexyphenidyl = fcol1.checkbox("Trihexyphenidyl")
    clonidine = fcol2.checkbox("Clonidine")
    sertralina = fcol3.checkbox("Sertralina")
    tramadol = fcol1.checkbox("Tramadol")
    mexazolam = fcol2.checkbox("Mexazolam", help="Sedoxil")
    trospium = fcol3.checkbox("Trospium")
    alcoolico = fcol1.slider("Alcoolico", 0, 1, 0)

    # Guardar o dicionário numa variável
    user_data = {
        "proveniencia": proveniencia,
        "idade": idade,
        "gender": convertGenderToInt(gender),
        "tempo": tempo,
        "glicose": glicose,
        "sodio": sodio,
        "ureia": ureia,
        "creatinina": creatinina,
        "pcr": pcr,
        "ph": ph,
        "ca": ca,
        "ureia": ureia,
        "co2": co2,
        "hco3": hco3,
        "antidislipidemicos": antidislipidemicos,
        "alprazolam": convertCheckboxToInt(alprazolam),
        "captopril": convertCheckboxToInt(captopril),
        "codeine": convertCheckboxToInt(codeine),
        "desloratadine": convertCheckboxToInt(desloratadine),
        "diazepam": convertCheckboxToInt(diazepam),
        "lorazepam": convertCheckboxToInt(lorazepam),
        "digoxin": convertCheckboxToInt(digoxin),
        "dipyridamole": convertCheckboxToInt(dipyridamole),
        "furosemide": convertCheckboxToInt(furosemide),
        "fluvoxamine": convertCheckboxToInt(fluvoxamine),
        "haloperidol": convertCheckboxToInt(haloperidol),
        "hydrocortisone": convertCheckboxToInt(hydrocortisone),
        "iloperidone": convertCheckboxToInt(iloperidone),
        "morphine": convertCheckboxToInt(morphine),
        "nifedipine": convertCheckboxToInt(nifedipine),
        "paliperidone": convertCheckboxToInt(paliperidone),
        "prednisone": convertCheckboxToInt(prednisone),
        "ranitidine": convertCheckboxToInt(ranitidine),
        "risperidone": convertCheckboxToInt(risperidone),
        "trazodone": convertCheckboxToInt(trazodone),
        "venlafaxine": convertCheckboxToInt(venlafaxine),
        "warfarin": convertCheckboxToInt(warfarin),
        "amitriptyline": convertCheckboxToInt(amitriptyline),
        "hydroxyzine": convertCheckboxToInt(hydroxyzine),
        "paroxetine": convertCheckboxToInt(paroxetine),
        "quetiapine": convertCheckboxToInt(quetiapine),
        "scopolamine": convertCheckboxToInt(scopolamine),
        "trihexyphenidyl": convertCheckboxToInt(trihexyphenidyl),
        "clonidine": convertCheckboxToInt(clonidine),
        "sertralina": convertCheckboxToInt(sertralina),
        "tramadol": convertCheckboxToInt(tramadol),
        "mexazolam": convertCheckboxToInt(mexazolam),
        "trospium": convertCheckboxToInt(trospium),
        "alcoolico": alcoolico,
        
    }

    # Transformar os dados num dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features