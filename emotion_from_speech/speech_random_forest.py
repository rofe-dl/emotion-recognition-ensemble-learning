from process_dataset.speech_features import get_speech_features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = get_speech_features()
# df = pd.DataFrame(features, columns=['MFCCS, Chroma, Mel, Contrast, Tonnetz', 'Emotion'])

# features = df.drop('Emotion', axis=1)
# features['MFCCS, Chroma, Mel, Contrast, Tonnetz'] = 1

# nsamples, nx, ny = features.shape
# features = features.reshape((nsamples, nx*ny))

# labels = df['Emotion']

# df['Length'] = df['MFCCS, Chroma, Mel, Contrast, Tonnetz'].apply(lambda x: len(x))
# print(df['Length'].unique())

x_train, x_test, y_train, y_test = train_test_split(np.array(data[0]), np.array(data[1]), random_state=42, test_size=0.1)

# rfc = MLPClassifier(**model_params)
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

results = rfc.predict(x_test)

accuracy = accuracy_score(y_test, results)

print(accuracy)