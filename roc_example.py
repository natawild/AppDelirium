
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, f1_score, precision_recall_curve
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('1 - Especificidade (FFP)')
    plt.ylabel('Sensibilidade (FVP)')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()
    
data_X, cls_lab = make_classification(n_samples=2100, n_classes=2, weights=[1,1], random_state=2)
train_X, test_X, train_y, test_y = train_test_split(data_X, cls_lab, test_size=0.5, random_state=2)

model =RandomForestClassifier()
model.fit(train_X, train_y)

prob = model.predict_proba(test_X)
prob = prob[:, 1]
fper, tper, thresholds = roc_curve(test_y, prob)
auc_score= roc_auc_score(test_y, prob)

print(f"Test ROC AUC  Score: {roc_auc_score(test_y, prob)}")
plot_roc_curve(fper, tper)




#create dataset with 5 predictor variables
X, y = datasets.make_classification(n_samples=1000,
                                    n_features=4,
                                    n_informative=3,
                                    n_redundant=1,
                                    random_state=0)

#split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=0)

#fit logistic regression model to dataset
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#use logistic regression model to make predictions
y_score = classifier.predict_proba(X_test)[:, 1]


#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='red')

#add axis labels to plot
ax.set_title('Curva Especificidade-Sensibilidade')
ax.set_ylabel('Especificidade')
ax.set_xlabel('Sensibilidade')

#display plot
plt.show()