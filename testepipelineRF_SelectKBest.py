from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)


seed = 100  # so that the result is reproducible


def setupChart(plt):
    plt.rcParams["figure.figsize"] = [7.00, 5.50]
    plt.rcParams["figure.autolayout"] = True


# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pd.read_csv(filename)
    # split into input (X) and output (y) variables
    X = data.drop("Delirium", axis=1)
    y = data["Delirium"]
    return X, y


setupChart(plt)

X, y = load_dataset("./dados_apos_p_processamento.csv")


X_train_des, X_test, y_train_des, y_test = train_test_split(
    X, y, test_size=0.36, random_state=45673
)


# Fazer overSample apenas dos dados de treino, pois se fizer de todos, pode acontecer de ter repetidos nos dados de teste
rus = ADASYN(random_state=32)
X_train, y_train = rus.fit_resample(X_train_des, y_train_des)
print("Tamanho dos dados de Treino apos ADASYN", X_train.shape, y_train.shape)

print("\nXTRAIN:\n", X_train)

# Feature selector
selector_k_best = SelectKBest(f_regression, k=15)

# Random forest classifier
classifier = RandomForestClassifier()

# Build the machine learning pipeline
pipeline_classifier = Pipeline([("selector", selector_k_best), ("rf", classifier)])

# We can set the parameters using the names we assigned earlier. For example, if we want to set 'k' to 6 in the
# feature selector and set 'n_estimators' in the Random Forest Classifier to 25, we can do it as shown below


# Training the classifier
pipeline_classifier.fit(X_train, y_train)

# Predict the output - recebe as variáveis independentes e calcula a variável dependente
pred = pipeline_classifier.predict(X_test)
print("\nPredictions:\n", pred)


print("\nScore:", pipeline_classifier.score(X_test, y_test))

print("\nScore:", accuracy_score(y_test, pred))


# y_test.to_csv('y_predict.csv')
# ypred = pd.read_csv('y_pred.csv')
# npred = pred.reshape(-1, 1)
# print(npred)
# print("\nYTEST:\n", y_test)

# teste para tentar resolver 2D array
# y_test.reset_index(drop=True, inplace=True)
# pd.drop(y_test.index[2])

"""

# Print score
print("\nScore:", pipeline_classifier.score(y_test, pred))

# Print the selected features chosen by the selector
features_status = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_status):
    if item:
        selected_features.append(count)
print("\nSelected features (0-indexed):", ', '.join([str(x) for x in selected_features]))

"""
