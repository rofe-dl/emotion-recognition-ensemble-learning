from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import string
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle

def get_data():

    df = pd.read_csv('data/text_data_clean.csv')
    df.drop(df.loc[(df['Emotion'] == 'xxx') | (df['Emotion'] == 'dis') | (df['Emotion'] == 'oth') | (df['Emotion'] == 'fea') | (df['Emotion'] == 'sur') | (df['Emotion'] == 'fru')].index, inplace = True)
    df.loc[(df['Emotion'] == 'exc'), 'Emotion'] = 'hap'

    df = shuffle(df, random_state=42)

    x = df["Cleaned"]
    y = df["Emotion"]

    return x, y


def get_train_test():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    cv = CountVectorizer(min_df=5, ngram_range=(1, 2))
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    return x_train, x_test, y_train, y_test