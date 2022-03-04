from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# feature selection
def select_features(X_train, y_train, X_test, k):
    # configure to select a subset of features
    fs = SelectKBest(score_func=f_classif, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    # Get columns names
    cols_names = fs.get_feature_names_out()
    print(cols_names)
    # get columns indexs
    cols_indexs = fs.get_support()
    return X_train_fs, X_test_fs, fs, cols_indexs


def select_k_features(X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the pipeline to evaluate
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    fs = SelectKBest(score_func=f_classif)
    pipeline = Pipeline(steps=[("anova", fs), ("lr", model)])
    # define the grid
    grid = dict()
    grid["anova__k"] = [i + 1 for i in range(X.shape[1])]
    # define the grid search
    search = GridSearchCV(pipeline, grid, scoring="accuracy", n_jobs=-1, cv=cv)
    # perform the search
    return search.fit(X, y)


def print_select_k_features(X, y, k):
    # create pipeline
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    fs = SelectKBest(score_func=f_classif, k=k)
    pipeline = Pipeline(steps=[("anova", fs), ("lr", model)])
    # evaluate the model
    scores = evaluate_model(X, y, pipeline)
    # results.append(scores)
    # summarize the results
    print(">%d %.3f (%.3f)" % (k, mean(scores), std(scores)))


def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    return scores
