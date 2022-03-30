import pandas as pd
from soundfile import SoundFile
import numpy as np
import librosa
import pickle

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

def get_speech_features():
    with open('data/speech_features_hstacked_iemocap_cleaned.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return features

def make_speech_features():
    df = pd.read_csv(config['Dataset']['dataset_details_location'])
    df.loc[(df['emotion'] == 'exc'), 'emotion'] = 'hap'
    df.drop(df.loc[(df['emotion'] == 'xxx') | (df['emotion'] == 'dis') | (df['emotion'] == 'oth')].index, inplace = True)

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

# def _extract_features(file_name):

#     with SoundFile(file_name) as sound_file:
#         audio = sound_file.read(dtype="float32")
#         sample_rate = sound_file.samplerate

#         stft = np.abs(librosa.stft(audio))

#         mfccs = np.array(np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0))
#         chroma = np.array(np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0))
#         mel = np.array(np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0))
#         contrast = np.array(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0))
#         tonnetz = np.array(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T,axis=0))

#         # result = [mfccs, chroma, mel, contrast, tonnetz]

#     return mfccs, chroma, mel, contrast, tonnetz
#     # return result

# def make_speech_features():

#     df = pd.read_csv(config['Dataset']['dataset_details_location'])
#     df.loc[(df['emotion'] == 'exc'), 'emotion'] = 'hap'
#     df.drop(df.loc[(df['emotion'] == 'xxx') | (df['emotion'] == 'dis') | (df['emotion'] == 'oth')].index, inplace = True)

#     file_list = df['path'].tolist()
#     emotions = df['emotion'].tolist()

#     mfccss, chromas, mels, contrasts, tonnetzs, y = ([] for i in range(6))
    

#     for index, file_name in enumerate(file_list):
#         speech_features = _extract_features(config['Dataset']['iemocap_dataset_location'] + file_name)

#         mfccss.append(speech_features[0])
#         chromas.append(speech_features[1])
#         mels.append(speech_features[2])
#         contrasts.append(speech_features[3])
#         tonnetzs.append(speech_features[4])

#         # X.append(speech_features)
#         y.append(emotions[index])
#         print('On file number ', index + 1, '/7527')

#     # features = (X, y)
#     features = (mfccss, chromas, mels, contrasts, tonnetzs, y)
#     with open('speech_features_separated_iemocap.pkl', 'wb') as f:
#         pickle.dump(features, f)

# make_speech_features()
# get_speech_features()