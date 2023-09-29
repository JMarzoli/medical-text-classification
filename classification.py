# Classification models
import os.path
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

path = 'models'
lr_path = path + '/logistic-regression'
nb_path = path + '/naive-bayes'
svm_path = path + '/support-vector-machine'
rf_path = path + '/random-forest'
dt_path = path + '/decision-tree'

# if os.path.exists(lr_path):
#     os.remove(lr_path)
# if os.path.exists(nb_path):
#     os.remove(nb_path)
# if os.path.exists(svm_path):
#     os.remove(svm_path)
# if os.path.exists(rf_path):
#     os.remove(rf_path)
# if os.path.exists(dt_path):
#     os.remove(dt_path)


def logistic_regression(X_train, X_test, y_train):
    if not os.path.exists(lr_path):
        clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=1).fit(X_train, y_train)
        with open(lr_path, 'wb') as file:
            pickle.dump(clf, file)
    else:
        with open(lr_path, 'rb') as file:
            clf = pickle.load(file)
    return clf.predict(X_test), clf.predict_proba(X_test)


def naive_bayes(X_train, X_test, y_train):
    if not os.path.exists(nb_path):
        gnb = GaussianNB().fit(X_train, y_train)
        with open(nb_path, 'wb') as file:
            pickle.dump(gnb, file)
    else:
        with open(nb_path, 'rb') as file:
            gnb = pickle.load(file)
    return gnb.predict(X_test), gnb.predict_proba(X_test)


def support_vector_machine(X_train, X_test, y_train):
    if not os.path.exists(svm_path):
        # creating a Logistic Regression Model and training it
        svc = svm.SVC(probability=True).fit(X_train, y_train)
        with open(svm_path, 'wb') as file:
            pickle.dump(svc, file)
    else:
        with open(svm_path, 'rb') as file:
            svc = pickle.load(file)
    return svc.predict(X_test), svc.predict_proba(X_test)


def random_forest(X_train, X_test, y_train):
    if not os.path.exists(rf_path):
        rf = RandomForestClassifier().fit(X_train, y_train)
        with open(rf_path, 'wb') as file:
            pickle.dump(rf, file)
    else:
        with open(rf_path, 'rb') as file:
            rf = pickle.load(file)
    return rf.predict(X_test), rf.predict_proba(X_test)


def decision_tree(X_train, X_test, y_train):
    if not os.path.exists(dt_path):
        dt = DecisionTreeClassifier().fit(X_train, y_train)
        with open(dt_path, 'wb') as file:
            pickle.dump(dt, file)
    else:
        with open(dt_path, 'rb') as file:
            dt = pickle.load(file)
    return dt.predict(X_test), dt.predict_proba(X_test)


# Class for easily storing metrics of models
class ModelReport:
    accuracy = None  # number of correctly classified / number of all instance
    recall = None  # % of correctly labelled positive instance out of all positive labelled instance (TP/TP+FN)
    precision = None  # of correctly labelled positive instance out of all positive instance (TP/TP+FP)
    f1 = None  # combination of precision and recall (2/(1/precision)+(1(recall))
    support = None
    roc = None
    auc = None

    def __init__(self, y_prediction, y_test):
        self.y_prediction = y_prediction
        self.y_test = y_test
        self.evaluate_metrics()

    def evaluate_metrics(self):
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_prediction)
        self.recall = metrics.recall_score(self.y_test, self.y_prediction, average='micro')
        self.precision = metrics.precision_score(self.y_test, self.y_prediction, average='micro')
        self.f1 = metrics.f1_score(self.y_test, self.y_prediction, average='micro')
        self.support = None  # TODO
        # self.roc = metrics.roc_curve(self.y_test, self.y_prediction) # TODO restricted to binary classification
        # self.auc = metrics.auc(self.y_test, self.y_prediction) # TODO remove bug
