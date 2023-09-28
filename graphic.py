# Logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataframe_image as dfi
import numpy as np
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import gc
from nltk import word_tokenize
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer

from classification import ModelReport

log_path = 'log/'
dataset_path = log_path + 'dataset/'
vectorization_path = log_path + 'text-representation/'
models_path = log_path + 'classification-models/'


np.set_printoptions(threshold=np.inf)


def log_df_info(df: pd.DataFrame):
    new_path = log_path + "df-info.txt"
    file = open(new_path, 'w')
    file.write('Dataframe shape: ' + str(df.shape) + '\n')
    file.write('Mean length of the transcription in corpus: ' + str(__mean_length_transcriptions(df)) + '\n')


def plot_df(df: pd.DataFrame):
    new_path = dataset_path + 'processed-dataframe.png'
    dfi.export(df, new_path, max_rows=30)


def log_classes(df: pd.DataFrame):
    new_path = log_path + 'classes.txt'
    classes = df.groupby(df['medical_specialty'])
    f = open(new_path, 'w')
    for cat_name, data_category in classes:
        string = str(cat_name).strip() + ': ' + str(len(data_category)) + '\n'
        f.write(string)
    f.close()


def plot_classes(df: pd.DataFrame):
    new_path = dataset_path + 'classes.png'
    plt.figure(figsize=(14, 10))
    sns.countplot(y='medical_specialty', data=df)
    plt.savefig(new_path)
    plt.clf()


def log_features(features):
    new_path = log_path + 'features.txt'
    file = open(new_path, 'w')
    file.write('Number of features: ' + str(len(features)) + '\n')
    file.write(str(features))
    file.close()


def __mean_length_transcriptions(df):
    array = df['transcription'].apply(lambda n: len(str(n).split()))
    return np.mean(array)


def log_model_reports(predictions, y_test, labels, txt: bool = False, csv: bool = False):
    if not txt and not csv:
        return
    for i in range(0, len(predictions)):
        file_name = models_path + str(predictions[i][1]) + '/' + str(predictions[i][1])
        y_prediction = predictions[i][0]
        if txt:
            cr = classification_report(y_test, y_prediction, labels=labels)
            txt_path = file_name + '.txt'
            with open(txt_path, 'w') as txt:
                txt.write(str(cr))
                txt.close()
        if csv:
            cr = classification_report(y_test, y_prediction, labels=labels, output_dict=True)
            csv_path = file_name + '.csv'
            pd.DataFrame(cr).transpose().to_csv(csv_path)


def log_report(report: ModelReport, file_name):
    table = PrettyTable()
    table.field_names = ['Metric', 'Value']
    table.add_row(["Accuracy", report.accuracy])
    table.add_row(["Recall", report.recall])
    table.add_row(["Precision", report.precision])
    table.add_row(["F1", report.f1])
    table.add_row(["Roc", report.roc])
    table.add_row(["Auc", report.auc])
    new_path = models_path + '/' + file_name
    with open(new_path, 'w') as file:
        file.write(table.get_string())
        file.close()


def plot_tfidf_matrix(tfidf_matrix, labels, filename):
    new_path = vectorization_path + 'tfidf-matrix/' + filename + '.png'
    gc.collect()
    tsne_results = TSNE(n_components=2, init='random', random_state=0, perplexity=40).fit_transform(tfidf_matrix)
    plt.figure(figsize=(20, 10))
    palette = sns.hls_palette(12, l=.3, s=.9)
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
        alpha=0.3
    )
    plt.savefig(new_path)
    plt.clf()


def plot_confusion_matrix(y_test, y_prediction, labels, file_name):
    new_path = vectorization_path + 'confusion-matrix/' + file_name + '.png'
    cm = confusion_matrix(y_true=y_test, y_pred=y_prediction, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig(new_path)
    plt.clf()


def plot_confusion_matrices(predictions, y_test, labels):
    for i in range(0, len(predictions)):
        y_prediction = predictions[i][0]
        new_path = models_path + str(predictions[i][1]) + '/' + str(predictions[i][1]) + '.png'
        cm = confusion_matrix(y_true=y_test, y_pred=y_prediction, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig(new_path)
        plt.clf()


def log_features_examples(df, number):
    new_path = log_path + 'features-example.txt'
    file = open(new_path, 'w')
    for i in range(1, number):
        text = df.iloc[i]['transcription']
        features = word_tokenize(text)
        file.write('Transcription ' + str(i) + ' [' + str(len(features)) + ']: ' + str(features) + '\n')
    file.close()


def plot_roc_curves(predictions, y_train, y_test, classes):
    roc_path = 'log/classification-models/'
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot = label_binarizer.transform(y_test)
    for i in range(0, len(predictions)):
        model_path = roc_path + str(predictions[i][1])
        y_score = predictions[i][0]
        for class_of_interest in classes:
            class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
            RocCurveDisplay.from_estimator(
                y_onehot[:, class_id],
                y=y_score,
                name=f"{class_of_interest} vs the rest",
                color="darkorange",
                plot_chance_level=True,
            )
            plt.axis("square")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("One-vs-Rest ROC curves:\n" + str(class_of_interest) + " vs All")
            plt.legend()
            final_path = model_path + str(class_of_interest).lower().replace(" ", "").replace("/", "-")
            plt.savefig(final_path)
            plt.clf()



