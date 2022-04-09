import pandas as pd
from soundfile import SoundFile
import numpy as np
import librosa
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle

def _get_speech_features():
    with open('data/speech_features.pkl', 'rb') as f:
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
    # df.loc[(df['emotion'] == 'fru'), 'emotion'] = 'ang'
    df.drop(df.loc[(df['emotion'] == 'xxx') | (df['emotion'] == 'dis') | (df['emotion'] == 'oth') | (df['emotion'] == 'fea') | (df['emotion'] == 'sur') | (df['emotion'] == 'fru')].index, inplace = True)

    df = shuffle(df, random_state=42)

    file_list = df['path'].tolist()
    emotions = df['emotion'].tolist()

    X, y = [], []

    for index, file_name in enumerate(file_list):
        speech_features = _extract_features('data/IEMOCAP_dataset/' + file_name)

        X.append(speech_features)
        y.append(emotions[index])
        print('On file number ', index + 1, '/', len(file_list))

    features = (X, y)

    with open('data/speech_features.pkl', 'wb') as f:
        pickle.dump(features, f)

def _extract_features(file_name):

    result = np.array([])

    with SoundFile(file_name) as sound_file:
        audio = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        stft = np.abs(librosa.stft(audio))
        result = np.array([])

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        pitches = np.mean(librosa.piptrack(y=audio, sr=sample_rate)[0].T, axis=0)

        rms = librosa.feature.rms(y=audio)[0]
        rms = np.array([np.mean(rms), np.median(rms), np.max(rms), np.min(rms), np.std(rms), np.average(rms), np.var(rms)])

        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        zcr = np.array([np.mean(zcr), np.median(zcr), np.max(zcr), np.min(zcr), np.std(zcr), np.average(zcr), np.var(zcr)])

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio,sr=sample_rate)[0]
        spectral_rolloff = np.array([np.mean(spectral_rolloff), np.median(spectral_rolloff), np.max(spectral_rolloff), np.min(spectral_rolloff), np.std(spectral_rolloff), np.average(spectral_rolloff), np.var(spectral_rolloff)])

        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        spectral_flux = np.array([np.mean(spectral_flux), np.median(spectral_flux), np.max(spectral_flux), np.min(spectral_flux), np.std(spectral_flux), np.average(spectral_flux), np.var(spectral_flux)])

        result = np.hstack((result, mfccs, chroma, mel, contrast, pitches, rms, zcr, spectral_rolloff, spectral_flux))

    return result

if __name__ == '__main__':

    make_speech_features()