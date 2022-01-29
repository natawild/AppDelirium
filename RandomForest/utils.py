import matplotlib.pyplot as plt
import itertools
import numpy as np

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

from sklearn.ensemble import RandomForestClassifier

def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Greens
):  # can change color
    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=20)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=18,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("True label", size=16)
    plt.xlabel("Predicted label", size=16)
    plt.show()



#Feature importance 
def get_index_value ( arr, features_support, column_names ):
    features = []
    for index, value in enumerate(arr): 
        if features_support[index]:
            pair = [ arr[index], column_names[index]]
            features.append(pair)
    return features


def run_random_forest (X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: \n ', accuracy_score(y_test,y_pred))
    print('Recall: \n ', recall_score(y_test,y_pred))
    print('Precision: \n ', precision_score(y_test,y_pred))

