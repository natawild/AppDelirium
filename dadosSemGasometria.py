import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

def get_dataset(name):
	dadosSemGasometria = None
	if name == 'Sem Gasometria':
        dadosComGasometria = pd.read_csv('./DeliriumsemGasometria.csv')
        X = dadosComGasometria.iloc[:, 0:48].values
        y = dadosComGasometria.iloc[:, -1].values
    return X, y

    