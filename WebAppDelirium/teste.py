import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

dadosComGasometria = pd.read_csv(
    "/Users/user/Documents/GitHub/AppDelirium/DeliriumcomGasometria.csv"
)

print(dadosComGasometria.describe())
