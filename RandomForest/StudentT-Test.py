import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import t

seed(1)

data = pd.read_csv("dadosLimposNumericos.csv")

print(data.head())

t_stat, p = ttest_ind(data["Genero"], data["Delirium"])
print(f"T-Student Genero ={t_stat}, p={p}")
stat, p = ttest_rel(data["Genero"], data["Delirium"])
print(f"T-Statistics Genero ={stat}, p={p}")


t_stat, p = ttest_ind(data["Idade"], data["Delirium"])
print(f"T-Student _Idade ={t_stat}, p={p}")
stat, p = ttest_rel(data["Idade"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["Interna_Dias"], data["Delirium"])
print(f"T-Student Interna_Dias ={t_stat}, p={p}")
stat, p = ttest_rel(data["Interna_Dias"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["SIRS"], data["Delirium"])
print(f"T-Student SIRS ={t_stat}, p={p}")
stat, p = ttest_rel(data["SIRS"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["Glicose"], data["Delirium"])
print(f"T-Student Glicose ={t_stat}, p={p}")
stat, p = ttest_rel(data["Glicose"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["Sodio"], data["Delirium"])
print(f"T-Student Sodio ={t_stat}, p={p}")
stat, p = ttest_rel(data["Sodio"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["Ureia"], data["Delirium"])
print(f"T-Student Ureia ={t_stat}, p={p}")
stat, p = ttest_rel(data["Ureia"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["Creatinina"], data["Delirium"])
print(f"T-Student Creatinina ={t_stat}, p={p}")
stat, p = ttest_rel(data["Creatinina"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["pH"], data["Delirium"])
print(f"T-Student pH ={t_stat}, p={p}")
stat, p = ttest_rel(data["pH"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["Ca_ionizado"], data["Delirium"])
print(f"T-Student Ca_ionizado ={t_stat}, p={p}")
stat, p = ttest_rel(data["Ca_ionizado"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["pCO2"], data["Delirium"])
print(f"T-Student pCO2 ={t_stat}, p={p}")
stat, p = ttest_rel(data["pCO2"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["pO2"], data["Delirium"])
print(f"T-Student pO2 ={t_stat}, p={p}")
stat, p = ttest_rel(data["pO2"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["HCO3"], data["Delirium"])
print(f"T-Student HCO3 ={t_stat}, p={p}")
stat, p = ttest_rel(data["HCO3"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


t_stat, p = ttest_ind(data["PCR"], data["Delirium"])
print(f"T-Student _PCR ={t_stat}, p={p}")
stat, p = ttest_rel(data["PCR"], data["Delirium"])
print(f"T-Statistics={stat}, p={p}")


def plot_distribution(inp):
    plt.figure()
    ax = sns.displot(inp)
    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
    _, max_ = plt.ylim()
    plt.text(
        inp.mean() + inp.mean(),
        max_ - max_,
        "Mean: {:.2f}".format(inp.mean()),
    )
    return plt.figure


pcr = data["PCR"]
delirium = data["Delirium"]

plot_distribution(pcr)
plt.show()

plot_distribution(delirium)
plt.show()


plt.figure()
ax1 = sns.displot(pcr)
ax2 = sns.displot(delirium)
plt.axvline(np.mean(pcr), color="b", linestyle="dashed", linewidth=5)
plt.axvline(np.mean(delirium), color="orange", linestyle="dashed", linewidth=5)
plt.show()


def compare_2_groups(arr_1, arr_2, alpha, sample_size):
    stat, p = ttest_ind(arr_1, arr_2)
    print("Statistics=%.3f, p=%.3f" % (stat, p))
    if p > alpha:
        print("Same distributions (fail to reject H0)")
    else:
        print("Different distributions (reject H0)")


sample_size = 100
pcr_sampled = np.random.choice(pcr, sample_size)
deli_sampled = np.random.choice(delirium, sample_size)
compare_2_groups(pcr_sampled, deli_sampled, 0.05, sample_size)

t_stat, p = ttest_ind(pcr, delirium)
print(t_stat, p)
