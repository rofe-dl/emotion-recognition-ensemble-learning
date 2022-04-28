import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv('data/text_data_clean.csv')
df.drop(df.loc[(df['Emotion'] == 'xxx') | (df['Emotion'] == 'dis') | (df['Emotion'] == 'oth') | (df['Emotion'] == 'fea') | (df['Emotion'] == 'sur') | (df['Emotion'] == 'fru')].index, inplace = True)
df.loc[(df['Emotion'] == 'exc'), 'Emotion'] = 'hap'

df = shuffle(df, random_state=42)

x = df["Cleaned"]
cv = CountVectorizer(min_df=5, ngram_range=(1, 2))
x = cv.fit_transform(x)
y = df["Emotion"]

print(x.shape)