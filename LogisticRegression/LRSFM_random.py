import streamlit as st
import numpy as np
import pandas as pd
import utils
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    f_classif,
    RFE,
    RFECV,
    f_regression,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
    KFold,
    RandomizedSearchCV
)

from matplotlib import pyplot
from numpy import mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.tree import plot_tree
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
import mlxtend
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_recall_curve, 
    auc
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import make_column_transformer
from numpy import asarray
import missingno as msno
from imblearn.under_sampling import RandomUnderSampler, NearMiss, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
import itertools
from sklearn.utils.fixes import loguniform


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Greens
):  # can change color
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=13)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=12,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("Valor Real", size=12)
    plt.xlabel("Valor Previsto", size=12)
    plt.show()
    

# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop("Delirium", axis=1)
    y = data["Delirium"]
    return X, y


###############################################################################
#             Train model with all features                                   #
###############################################################################


X, y = load_dataset("./dados_apos_p_processamento.csv")
#X, y = load_dataset("./resnovos3.csv")

print("heelelelelelel",X.shape)



X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673, stratify=y
)

dados = X_test
df = pd.DataFrame(dados).to_csv("dadosTeste.csv")

ytest=y_test
df = pd.DataFrame(ytest).to_csv("YdadosTeste.csv")


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=100)
# rus = RandomUnderSampler(random_state = 32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)


# Create a logistic regression classifier
clf = LogisticRegression(C= 1.0,
 class_weight= None,
 dual= False,
 fit_intercept= True,
 intercept_scaling= 1,
 l1_ratio= None,
 max_iter= 100,
 multi_class= 'ovr',
 n_jobs= None,
 penalty= 'l2',
 random_state= 78787879,
 solver= 'lbfgs',
 tol= 0.0001,
 verbose= 0,
 warm_start= False)

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and importance of each feature
for feature in zip(X_train.columns, clf.coef_):
    print(feature)



# Apply The Full Featured Classifier To The Test Data
y_pred_all = clf.predict(X_test)
yhat = clf.predict_proba(X_test)
yhat_positive = yhat[:, 1]

# View The Accuracy
acc = round(accuracy_score(y_test, y_pred_all), 4) * 100
print("The accuracy of the model is:\n", acc)

recall = round(recall_score(y_test, y_pred_all), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred_all), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred_all), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred_all)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, yhat_positive)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred_all)
print(matrix)

#--------------------precision_recall_curve---------------

# predict probabilities
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('RF: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

print('intercept ', clf.intercept_[0])
print('classes', clf.classes_)
print('value of coef_:',clf.coef_[0])
# Print slope and intercept
print('Intercept (Beta 0): ', clf.intercept_)
print('Slope (Beta 1): ', clf.coef_)


# Look at parameters used by current model
print('Parameters currently in use:\n')
pprint(clf.get_params())

# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred_all),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Matriz de confusão sem seleção de variáveis (RL_SFM0.09)",
)



#--------------------precision_recall_curve---------------


# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, label='AUC ' + str(lr_auc))
# axis labels
pyplot.title('Curva Especificidade-Sensibilidade sem seleção de variáveis (RL)')
pyplot.xlabel('Sensibilidade')
pyplot.ylabel('Especificidade')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# plot roc curve
plt.plot(fpr, tpr, color='red', label='AUC ' + str(auc_score))
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
# axis labels
plt.xlabel('1 - Especificidade (FFP)')
plt.ylabel('Sensibilidade (FVP)')
plt.title('Curva ROC sem seleção de variáveis (RL)')
# show the legend
plt.legend()
plt.show()


###############################################################################
#             Select most important features with  SFM                    #
###############################################################################


# Feature Selection 
sel = SelectFromModel(
    clf,
    threshold=1.5, 
)
print("Parametros SelectFromModel\n \n \n ",sel)

sel = sel.fit(X_train, y_train)


print("Dictonary with metrics")
#http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-4-plotting-the-results
pd.DataFrame.from_dict(sel.get_support()).T.to_csv("RL_SFM15_dict.csv")


logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print("LLOOOGGGGIIIITTT",result.summary2())


###############################################################################
#             Train model with only the most important features               #
###############################################################################


# Create A Data Subset With Only The Most Important Features
X_train_rfc = sel.transform(X_train)
X_test_rfc = sel.transform(X_test)

#print('X TRAIN',X_train_rfc.shape)
#print('Y TRAIN',y_train.shape)
#print('X Test',X_test_rfc.shape)
#print('Y TEST',y_test.shape)

#create and Train A New Logistic Regression Classifier Using Only Most Important Features
clf = LogisticRegression()
clf.fit(X_train_rfc,y_train)

#predict results 
y_pred = clf.predict(X_test_rfc)
print("###########################\n",y_pred)

#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print("The accuracy of the model is:\n", acc)

# predict probabilities
yhat = clf.predict_proba(X_test_rfc)
#print("#######YHAT: ", yhat)

# retrieve the probabilities for the positive class
yhat_positive = yhat[:, 1]
#print("yhat_positive: ", yhat_positive)

# calculate and print AUROC
roc_auc = roc_auc_score(y_test, yhat_positive)
print('ROC CURVE ---- : %.3f' % roc_auc)

recall = round(recall_score(y_test, y_pred), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred)
print("Classification report\n", c_r)

auc_score = roc_auc_score(y_test, yhat_positive)
print("ROC AUC \n", auc_score)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)


# predict probabilities
lr_probs = clf.predict_proba(X_test_rfc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = clf.predict(X_test_rfc)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('RL: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))




#--------------------------PLOT FEATURE IMPORTANCES------------------------



features_name = sel.get_feature_names_out()
#print('indices das features selecionadas: ', sfs.k_feature_idx_)
print('Length Features selected: \n',len(features_name))
print("Features selected:\n ", features_name)

print('intercept \n', clf.intercept_[0])
print('classes \n', clf.classes_)
print('value of coef_:\n',clf.coef_[0])

# Print slope and intercept
print('Intercept (Beta 0): ', clf.intercept_)
print('Slope (Beta 1): ', clf.coef_)
'''
print(pd.DataFrame({'coeff': clf.coef_[0]}, 
             index=X_train_rfc.columns))
'''


logit_model=sm.Logit(y_train,X_train_rfc)
result=logit_model.fit()
print("LLOOOGGGT",result.summary2())


f_i = list(zip(features_name,clf.coef_[0]))
print('qqqqqqqqqqqq',f_i)
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
# axis labels
plt.xlabel('Importância das variáveis')
plt.ylabel('Variáveis')
plt.title('Variáveis selecionadas e respetiva importância (RL_SFM1.5)')
plt.show()





# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Matriz de confusão após seleção de variáveis (RL_SFM0.09)",
)


# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# plot roc curve
plt.plot(fpr, tpr, color='red', label='AUC ' + str(roc_auc))
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
# axis labels
plt.xlabel('1 - Especificidade (FFP)')
plt.ylabel('Sensibilidade (FVP)')
plt.title('Curva ROC das variáveis selecionadas (RL_SFM0.09)')
# show the legend
plt.legend()
plt.show()
    



#--------------------------precision_recall_curve------------------------

# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, label='AUC ' + str(lr_auc))
# axis labels
pyplot.title('Curva Especificidade-Sensibilidade após seleção de variáveis (RL_SFM0.09)')
pyplot.xlabel('Sensibilidade')
pyplot.ylabel('Especificidade')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(clf.get_params())



###############################################################################
#                             Tunning model                                                   #
###############################################################################


# create a grid of parameters for the model to pick and train
penalty = ["l1", "l2", "elasticnet","none"]
#penalty = ["elasticnet"]
#C = [loguniform(1e0, 1e3)]
C = [1.0, 1.001, 1.01]
#C = [1.0, 1.001, 1.01, 1.0009]
#max_iter = [75, 80, 90, 100, 150, 500]
max_iter = [int(x) for x in np.arange(start=2, stop=500, step=5)]
solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
#solver = ["saga", "lbfgs"]
warm_start =["True","False"]
#multi_class=["auto", "ovr", "multinomial"]
multi_class=[ "ovr"]
#l1_ratio = [int(x) for x in np.arange(start=0, stop=1, step=0.1)]
l1_ratio = [0.0000001, 0.001, 0, 0.00001, 0.1]


param_grid = dict(penalty=penalty,
                  C=C,
                  solver=solver, 
                  max_iter= max_iter,
                  multi_class = multi_class,
                  l1_ratio = l1_ratio,
                  warm_start= warm_start)



#-------------------------------------------RandomizedSearchCV--------------------------------------------------#
# Create random search model and fit the data
random_grid = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_grid,
    n_iter=100,
    cv=RepeatedStratifiedKFold(),
    scoring="roc_auc",
)

'''
#-------------------------------------------GridSearchCV--------------------------------------------------#

grid = GridSearchCV(estimator=clf,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv = 5, 
                    verbose=0,
                    n_jobs=-1)

'''

grid_result = random_grid.fit(X_train_rfc, y_train)

# print winning set of hyperparameters
print("Best set of hyperparameters \n ",grid_result.best_estimator_.get_params())
print("Best score: \n",random_grid.best_score_)
print('Best params: \n', grid_result.best_params_)



# Use the best model after tuning
best_model = grid_result.best_estimator_


best_model.fit(X_train_rfc, y_train)

y_pred_best_model = best_model.predict(X_test_rfc)
train_rf_predictions = best_model.predict(X_train_rfc)
train_rf_probs = best_model.predict_proba(X_train_rfc)[:, 1]
rf_probs = best_model.predict_proba(X_test_rfc)[:, 1]


#calculate accuracy 
acc = round(accuracy_score(y_test, y_pred_best_model), 4) * 100
print("The accuracy of the model is:\n", acc)

recall = round(recall_score(y_test, y_pred_best_model), 4) * 100
print("The recall of the model is:\n", recall)

precision = round(precision_score(y_test, y_pred_best_model), 4) * 100
print("The precision of the model is:\n", precision)

f1 = round(f1_score(y_test, y_pred_best_model), 4) * 100
print("The f1 of the model is:\n", f1)

c_r = classification_report(y_test, y_pred_best_model)
print("Classification report\n", c_r)

matrix = confusion_matrix(y_test, y_pred_best_model)
print(matrix)

auc_score= roc_auc_score(y_test, rf_probs)
print("AUC SCORE:", auc_score)

#--------------------precision_recall_curve---------------

# predict probabilities
lr_probs = best_model.predict_proba(X_test_rfc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = best_model.predict(X_test_rfc)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

# summarize scores
print('RL: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))


# Plot Confusion matrix
plot_confusion_matrix(
    confusion_matrix(y_test, y_pred_best_model),
    classes=["0 - Sem Delirium", "1 - Delirium"],
    title="Matriz de confusão após afinação dos hiperparâmetros (RL_SFM0.09",
)


# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, label='AUC ' + str(lr_auc))
# axis labels
pyplot.title('Curva Especificidade-Sensibilidade após afinação (RL_SFM0.09)')
pyplot.xlabel('Sensibilidade')
pyplot.ylabel('Especificidade')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#------------------------------------ROC_curve---------------------------

# calculate inputs for the roc curve
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
# plot roc curve
plt.plot(fpr, tpr, color='red', label='AUC ' + str(auc_score))
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
# axis labels
plt.xlabel('1 - Especificidade (FFP)')
plt.ylabel('Sensibilidade (FVP)')
plt.title('Curva ROC após após afinação dos hiperparâmetros (RL_SFM0.09)')
# show the legend
plt.legend()
plt.show()
