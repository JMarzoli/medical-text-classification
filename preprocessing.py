# Preprocessing of the dataset
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import en_ner_bionlp13cg_md

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
sci_model = en_ner_bionlp13cg_md.load()

categories_to_remove = ['Surgery', 'SOAP / Chart / Progress Notes', 'Office Notes', 'Consult - History and Phy.',
                        'Emergency Room Reports', 'Discharge Summary', 'Pain Management', 'General Medicine']


def prepare_df(df: pd.DataFrame):
    df = __remove_useless_column(df)
    df = __remove_empty(df)
    df = __remove_categories_under(df=df, treshold=50)
    df = __remove_categories(df, categories_to_remove)
    df = __merge_categories(df, 'Neurology', 'Neurosurgery')
    df = __merge_categories(df, 'Urology', 'Nephrology')
    df['transcription'] = df['transcription'].apply(__clean_text)
    # df['transcription'] = df['transcription'].apply(__process_spacy_model)  # lower accuracy
    df['transcription'] = df['transcription'].apply(__remove_stopwords)
    df['transcription'] = df['transcription'].astype(str).apply(__lemmatize_text)
    df['transcription'] = df['transcription'].astype(str).apply(__stemming_text)
    df = df.drop(df[df['transcription'].isna()].index)
    return df


def __remove_useless_column(df):
    col = ['description', 'sample_name', 'keywords']
    df = df.drop(columns=col)
    return df


def __remove_empty(df):
    df = df.dropna(axis='index')
    return df


def __remove_categories_under(df, treshold):
    categories = df.groupby(df['medical_specialty'])
    df = categories.filter(lambda x: x.shape[0] > treshold)
    return df


def __clean_text(text):
    text = text.lower()  # lowercase text
    text = text.strip()  # get rid of leading/trailing whitespace
    text = re.compile('<.*?>').sub('', text)  # Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ',
                                                                  text)  # Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)  # Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]', ' ', text)  # [0-9] matches any digit (0 to 10000...)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)  # matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+', ' ',
                  text)  # \s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace
    return text


def __remove_stopwords(text):
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)


# bad results
def __stemming_text(text):
    text = [stemmer.stem(i) for i in word_tokenize(text) if i]
    return " ".join(text)


def __lemmatize_text(text):
    word_pos_tags = nltk.pos_tag(word_tokenize(text))  # Get position tags
    text = [lemmatizer.lemmatize(tag[0], __pos_tag(tag[1])) for idx, tag in
            enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(text)


# Helper function for lemmatizing function, wordnet treats some words as a noun and do not lemmatize it
# this function uses Part-of-Speech (POS) tags
def __pos_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def __process_spacy_model(text):
    wordlist = []
    doc = sci_model(text)
    for ent in doc.ents:
        wordlist.append(ent.text)
    return ' '.join(wordlist)


def __remove_categories(df, categories):
    df['medical_specialty'] = df['medical_specialty'].apply(
        lambda x: str.strip(x))
    for cat in categories:
        mask = df['medical_specialty'] == cat
        df = df[~mask]
    df = df.drop(df[df['transcription'].isna()].index)
    return df


def __merge_categories(df, receiving_cat, donor_cat):
    mask = df['medical_specialty'] == donor_cat
    df.loc[mask, 'medical_specialty'] = receiving_cat
    df = df.drop(df[df['transcription'].isna()].index)
    return df


