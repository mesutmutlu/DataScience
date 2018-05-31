from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
import time

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "d:/handsonml/"
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

def digit_csgd_classifier(x_train, y_train):
    # classifier for 5
    #y_train_d = (y_train == digit)  # True for all 5s, False for all other digits.

    sgd_clf = SGDClassifier(random_state=42, max_iter=5)
    sgd_clf.fit(x_train, y_train)

    return sgd_clf

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

if __name__ == "__main__":

    mnist = fetch_mldata('MNIST original')
    print(mnist)

    X, y = mnist["data"], mnist["target"]

    print(X.shape, y.shape)



    some_digit = X[36000]
    print(some_digit)
    some_digit_image = some_digit.reshape(28, 28)
    plot_digit(some_digit_image)
    print(y[36000])
    plt.show(block=False)
    time.sleep(1)
    plt.close()

    plt.figure(figsize=(9, 9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)
    save_fig("more_digits_plot")
    plt.show(block=False)
    time.sleep(1)
    plt.close()

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    y_train_5 = (y_train ==5)
    y_test_5 = (y_test == 5)

    sgd_clf = digit_csgd_classifier(X_train, y_train)
    print(sgd_clf.predict([some_digit]))

    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_5[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_5[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

    #print(X_test_fold.shape)

    from sklearn.base import BaseEstimator


    class Never5Classifier(BaseEstimator):
        def fit(self, X, y=None):
            pass

        def predict(self, X):
            return np.zeros((len(X), 1), dtype=bool)


    from sklearn.model_selection import cross_val_score
    never_5_clf = Never5Classifier()
    print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(y_train_5, y_train_pred))

    from sklearn.metrics import precision_score, recall_score

    print(precision_score(y_train_5, y_train_pred))
    print(recall_score(y_train_5, y_train_pred))

    from sklearn.metrics import f1_score
    print(f1_score(y_train_5, y_train_pred))
    print([some_digit])

    y_scores = sgd_clf.decision_function([some_digit])
    print(y_scores)

    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                 method="decision_function")

    print(y_scores.shape)
    print(y_scores)

    from sklearn.metrics import precision_recall_curve

    if y_scores.ndim == 2:
        y_scores = y_scores[:, 1]

    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xlim([-700000, 700000])
    save_fig("precision_recall_vs_threshold_plot")
    plt.show(block=False)
    time.sleep(1)
    plt.close()

    print((y_train_pred == (y_scores > 0)).all())

    y_train_pred_90 = (y_scores > 70000)

    print(precision_score(y_train_5, y_train_pred_90))

    print(recall_score(y_train_5, y_train_pred_90))

    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls)
    save_fig("precision_vs_recall_plot")
    plt.show(block=False)
    time.sleep(1)
    plt.close()

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    plot_roc_curve(fpr, tpr)
    plt.show()
    time.sleep(1)
    plt.close()

    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(y_train_5, y_scores)) # roc_auc_score(area under curve should approach to 1 and if it  approaches to 0.5, it means that the classifier is a random classifier

    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")

    y_scores_forest = y_probas_forest[:, 1]

    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    plt.plot(fpr, tpr, "b:", label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.show()
    time.sleep(1)
    plt.close()
    print("auc score of forest", roc_auc_score(y_train_5, y_scores_forest))

    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print("precision of forest", precision_score(y_train_5, y_train_pred_forest), "recall of forest", recall_score(y_train_5, y_train_pred_forest))