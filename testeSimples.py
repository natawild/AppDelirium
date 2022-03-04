# test classification dataset

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score


# define dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7
)
# summarize the dataset
print(X.shape, y.shape)
print(X)


# define the pipeline
steps = list()
steps.append(("scaler", MinMaxScaler()))
steps.append(("model", LogisticRegression()))
pipeline = Pipeline(steps=steps)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(pipeline, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
# report performance
print("Accuracy: %.3f (%.3f)" % (mean(scores) * 100, std(scores) * 100))
