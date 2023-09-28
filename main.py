# Main script
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN


import graphic
import preprocessing
import classification
from classification import Report

'''
1. Riportare risultati su drive (manca auc e roc)
2. Sistemare codice pre-processing (done) 
3. Valutare la fusione con altri dataset 
4. Studiare le metriche 
5. Valutare tecniche per improvement 
'''

df_path = 'datasets/processed.csv'
df = ''
max_features = 5000  # max number of features in tfidf matrix


# if os.path.exists(df_path):
#     os.remove(df_path)  # remove in production

if os.path.exists(df_path):
    df = pd.read_csv(df_path, sep=',', index_col=0)
else:
    df = pd.read_csv('datasets/original.csv', sep=',', index_col=0)
    df = preprocessing.prepare_df(df)
    df.to_csv('datasets/processed.csv')

graphic.log_df_info(df)
graphic.plot_df(df)
graphic.log_classes(df)
graphic.plot_classes(df)
graphic.log_features_examples(df, number=21)


labels = df['medical_specialty'].tolist()  # getting the classes to perform classification

# vectorize the corpus
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=0.75, min_df=5, use_idf=True, smooth_idf=True, sublinear_tf=True, max_features=max_features)
tfidf_matrix = vectorizer.fit_transform(df['transcription'].tolist())
tfidf_matrix = tfidf_matrix.toarray()  # if exception put under pca.fit
graphic.plot_tfidf_matrix(tfidf_matrix, labels, 'tfidf-matrix')
graphic.log_features(vectorizer.get_feature_names_out())

# applying linear dimensional reduction of features array
pca = PCA(n_components=0.95)
tfidf_matrix_reduced = pca.fit_transform(tfidf_matrix)
graphic.plot_tfidf_matrix(tfidf_matrix_reduced, labels, 'tfidf-matrix-pca')

# generating more samples for imbalanced class
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(tfidf_matrix_reduced, labels)
graphic.plot_tfidf_matrix(X_resampled, y_resampled, 'tfidf-matrix-smote')


# splitting the dataset into training and testing
category_list = df.medical_specialty.unique()
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, stratify=y_resampled, random_state=1)

# TODO refactor code below
# calculating predictions from all the models
lr_prediction = classification.logistic_regression(X_train, X_test, y_train)
nb_prediction = classification.naive_bayes(X_train, X_test, y_train)
svm_prediction = classification.support_vector_machine(X_train, X_test, y_train)
rf_prediction = classification.random_forest(X_train, X_test, y_train)
dt_prediction = classification.decision_tree(X_train, X_test, y_train)
# plotting confusion matrix of predictions
graphic.plot_confusion_matrix(y_test, lr_prediction, category_list, 'logistic-regression')
graphic.plot_confusion_matrix(y_test, nb_prediction, category_list, 'naive-bayes')
graphic.plot_confusion_matrix(y_test, svm_prediction, category_list, 'support-vector-machine')
graphic.plot_confusion_matrix(y_test, rf_prediction, category_list, 'random-forest')
graphic.plot_confusion_matrix(y_test, dt_prediction, category_list, 'decision-tree')
# logging the report of each prediction
graphic.log_report_txt(y_test, lr_prediction, category_list, 'logistic-regression')
graphic.log_report_txt(y_test, nb_prediction, category_list, 'naive-bayes')
graphic.log_report_txt(y_test, svm_prediction, category_list, 'support-vector-machine')
graphic.log_report_txt(y_test, rf_prediction, category_list, 'random-forest')
graphic.log_report_txt(y_test, dt_prediction, category_list, 'decision-tree')
graphic.log_report_csv(y_test, lr_prediction, category_list, 'logistic-regression')
graphic.log_report_csv(y_test, nb_prediction, category_list, 'naive-bayes')
graphic.log_report_csv(y_test, svm_prediction, category_list, 'support-vector-machine')
graphic.log_report_csv(y_test, rf_prediction, category_list, 'random-forest')
graphic.log_report_csv(y_test, dt_prediction, category_list, 'decision-tree')

# TODO get auc roc metrics




# predictions = []
# predictions.append([classification.logistic_regression(X_train, X_test, y_train), 'logistic-regression'])
# predictions.append([classification.naive_bayes(X_train, X_test, y_train), 'naive-bayes'])
# predictions.append([classification.support_vector_machine(X_train, X_test, y_train), 'support-vector-machine'])
# predictions.append([classification.random_forest(X_train, X_test, y_train), 'random-forest'])
# predictions.append([classification.decision_tree(X_train, X_test, y_train), 'decision-tree'])
#
# lr_report = Report(lr_prediction, y_test)
# nb_report = Report(nb_prediction, y_test)
# svm_report = Report(svm_prediction, y_test)
# rf_report = Report(rf_prediction, y_test)
# dt_report = Report(dt_prediction, y_test)
#
# graphic.log_report(lr_report, 'logistic-regression.txt')
# graphic.log_report(nb_report, 'naive-bayes.txt')
# graphic.log_report(svm_report, 'support-vector-machine.txt')
# graphic.log_report(rf_report, 'random-forest.txt')
# graphic.log_report(dt_report, 'decision-tree.txt')












