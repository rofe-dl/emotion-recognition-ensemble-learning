import pandas as pd
from soundfile import SoundFile
import numpy as np
import librosa
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def _get_speech_features():
    with open('data/ravdess.pkl', 'rb') as f:
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

def make_speech_features():
    df = pd.read_csv('iemocap_metadata.csv')
    df.loc[(df['emotion'] == 'exc'), 'emotion'] = 'hap'
    df.drop(df.loc[(df['emotion'] == 'xxx') | (df['emotion'] == 'dis') | (df['emotion'] == 'oth') | (df['emotion'] == 'fea') | (df['emotion'] == 'sur')].index, inplace = True)

    file_list = df['path'].tolist()
    emotions = df['emotion'].tolist()

    X, y = [], []

    for index, file_name in enumerate(file_list):
        speech_features = _extract_features('data/IEMOCAP_dataset/' + file_name)

        X.append(speech_features)
        y.append(emotions[index])
        print('On file number ', index + 1, '/', len(file_list))

    features = (X, y)

    with open('data/ravdess.pkl', 'wb') as f:
        pickle.dump(features, f)

def _extract_features(file_name):
    signal, sr = librosa.load(file_name, sr=22050)
    mfccs = np.mean(librosa.feature.mfcc(y=signal, n_fft=2048, hop_length=512, n_mfcc=100, sr=sr).T, axis=0)

    stft = np.abs(librosa.stft(signal))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T,axis=0)

    result = np.array([])
    result = np.hstack((result, mfccs, chroma, mel))
    # with SoundFile(file_name) as sound_file:
    #     audio = sound_file.read(dtype="float32")
    #     sample_rate = sound_file.samplerate

    #     stft = np.abs(librosa.stft(audio))
    #     result = np.array([])

    #     mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    #     # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    #     # mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
    #     # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    #     # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T,axis=0)

    #     # result = np.hstack((result, mfccs, chroma, mel, contrast, tonnetz))

    return result

if __name__ == '__main__':

    make_speech_features()