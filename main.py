# Main script
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN

import output
import preprocessing
import classification

'''
1. Automatizzare le metriche dei modelli in un dataframe 
2. Creare metodi per la comparazione dei risultati dei modelli 
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

# saving dataset info
output.log_df_info(df)
output.plot_df(df)
output.log_classes(df)
output.plot_classes(df)
output.log_features_examples(df, number=21)

labels = df['medical_specialty'].tolist()  # getting the classes to perform classification

# vectorize the corpus
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=0.75, min_df=5,
                             use_idf=True, smooth_idf=True, sublinear_tf=True, max_features=max_features)
tfidf_matrix = vectorizer.fit_transform(df['transcription'].tolist())
tfidf_matrix = tfidf_matrix.toarray()  # if exception put under pca.fit
output.plot_tfidf_matrix(tfidf_matrix, labels, 'tfidf-matrix')
output.log_features(vectorizer.get_feature_names_out())

# applying linear dimensional reduction of features array
pca = PCA(n_components=0.95)
tfidf_matrix_reduced = pca.fit_transform(tfidf_matrix)
output.plot_tfidf_matrix(tfidf_matrix_reduced, labels, 'tfidf-matrix-pca')

# generating more samples for imbalanced class
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(tfidf_matrix_reduced, labels)
output.plot_tfidf_matrix(X_resampled, y_resampled, 'tfidf-matrix-smote')

# splitting the dataset into training and testing
category_list = df.medical_specialty.unique()
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, stratify=y_resampled, random_state=1)

# creating a 2-dimensional array for saving the results of predictions
lr = classification.logistic_regression(X_train, X_test, y_train)
lr_prediction = [lr[0], lr[1], 'logistic-regression']
nb = classification.naive_bayes(X_train, X_test, y_train)
nb_prediction = [nb[0], nb[1], 'naive-bayes']
svm = classification.support_vector_machine(X_train, X_test, y_train)
svm_prediction = [svm[0], svm[1], 'support-vector-machine']
rf = classification.random_forest(X_train, X_test, y_train)
rf_prediction = [rf[0], rf[1], 'random-forest']
dt = classification.decision_tree(X_train, X_test, y_train)
dt_prediction = [dt[0], dt[1], 'decision-tree']
predictions = [lr_prediction, nb_prediction, svm_prediction, rf_prediction, dt_prediction]

# plotting the confusion matrices of predictions
output.plot_confusion_matrices(predictions, y_test, category_list)
# saving the model reports
output.log_model_reports(predictions, y_test, category_list, txt=True, csv=True)

# TODO get auc roc metrics

# graphic.plot_roc_curves(predictions, y_train, y_test, category_list)
output.log_roc_auc_score(predictions, y_test)

sys.exit()
