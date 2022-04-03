import pandas as pd
from soundfile import SoundFile
import numpy as np
import librosa
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

def _get_speech_features():
    with open('data/speech_features_hstacked_iemocap_cleaned.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return features

def get_data():
    data = _get_speech_features()
    x = np.array(data[0])
    y = np.array(data[1])

    x = MinMaxScaler().fit_transform(x)

    return x, y

def get_train_test():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    return x_train, x_test, y_train, y_test

def get_train_val_test():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    return x_train, x_val, x_test, y_train, y_val, y_test

def make_speech_features():
    df = pd.read_csv(config['Dataset']['dataset_details_location'])
    df.loc[(df['emotion'] == 'exc'), 'emotion'] = 'hap'
    df.drop(df.loc[(df['emotion'] == 'xxx') | (df['emotion'] == 'dis') | (df['emotion'] == 'oth') | (df['emotion'] == 'fea') | (df['emotion'] == 'sur')].index, inplace = True)

    file_list = df['path'].tolist()
    emotions = df['emotion'].tolist()

    # mfccss, chromas, mels, contrasts, tonnetzs, y= ([] for i in range(6))
    X, y = [], []

    for index, file_name in enumerate(file_list):
        speech_features = _extract_features(config['Dataset']['iemocap_dataset_location'] + file_name)

        # mfccss.append(speech_features[0])
        # chromas.append(speech_features[1])
        # mels.append(speech_features[2])
        # contrasts.append(speech_features[3])
        # tonnetzs.append(speech_features[4])

        X.append(speech_features)
        y.append(emotions[index])
        print('On file number ', index + 1, '/7527')

    features = (X, y)
    # features = (mfccss, chromas, mels, contrasts, tonnetzs, y)
    with open('speech_features_hstacked_iemocap_2.pkl', 'wb') as f:
        pickle.dump(features, f)

def _extract_features(file_name):

    with SoundFile(file_name) as sound_file:
        audio = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        stft = np.abs(librosa.stft(audio))
        result = np.array([])

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T,axis=0)

        result = np.hstack((result, mfccs, chroma, mel, contrast, tonnetz))

    # return mfccs, chroma, mel, contrast, tonnetz
    return result

# make_speech_features()
# get_speech_features()