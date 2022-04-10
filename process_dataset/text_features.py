from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import string
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle

def _get_text_features():
    with open('data/text_features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return features

def get_data():
    data = _get_text_features()
    x = data[0]
    y = data[1]

    return x, y

def get_train_test():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    return x_train, x_test, y_train, y_test

def make_text_features():
    df = pd.read_csv('data/text_data.csv')
    df.drop(df.loc[(df['Emotion'] == 'xxx') | (df['Emotion'] == 'dis') | (df['Emotion'] == 'oth') | (df['Emotion'] == 'fea') | (df['Emotion'] == 'sur') | (df['Emotion'] == 'fru')].index, inplace = True)
    df.loc[(df['Emotion'] == 'exc'), 'Emotion'] = 'hap'
    # df.loc[(df['Emotion'] == 'fru'), 'Emotion'] = 'ang'

    df = shuffle(df, random_state=42)

    x_features = df["Cleaned"]
    y_labels = df["Emotion"]

    cv = CountVectorizer(min_df=5, ngram_range=(1, 2))
    x_vectors = cv.fit_transform(x_features)

    tup = (x_vectors, y_labels)

    with open('data/text_features.pkl', 'wb') as f:
        pickle.dump(tup, f)

if __name__ == '__main__':

    make_text_features()