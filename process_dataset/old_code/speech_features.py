import pandas as pd
import numpy as np
import librosa
import os

def _extract_features(file_name):

    signal, sr = librosa.load(file_name, sr=22050)
    features = []

    mfccs = np.mean(librosa.feature.mfcc(y=signal, n_mfcc=40, sr=sr).T, axis=1)
    features.extend([np.mean(mfccs), np.median(mfccs), np.max(mfccs), np.min(mfccs), np.std(mfccs)])

    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=1)
    features.extend([np.mean(mel), np.median(mel), np.max(mel), np.min(mel), np.std(mel)])

    pitches = np.mean(librosa.piptrack(y=signal, sr=sr)[0].T, axis=1)
    features.extend([np.mean(pitches), np.median(pitches), np.max(pitches), np.min(pitches), np.std(pitches)])

    rms = librosa.feature.rms(y=signal)[0]
    features.extend([np.mean(rms), np.median(rms), np.max(rms), np.min(rms), np.std(rms)])

    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
    features.extend([np.mean(zcr), np.median(zcr), np.max(zcr), np.min(zcr), np.std(zcr)])
    
    chroma = np.mean(librosa.feature.chroma_stft(y=signal, n_chroma=40).T, axis=1)
    features.extend([np.mean(chroma), np.median(chroma), np.max(chroma), np.min(chroma), np.std(chroma)])

    spectral_flux = librosa.onset.onset_strength(y=signal, sr=sr)
    features.extend([np.mean(spectral_flux), np.median(spectral_flux), np.max(spectral_flux), np.min(spectral_flux), np.std(spectral_flux)])

    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    features.extend([np.mean(spectral_rolloff), np.median(spectral_rolloff), np.max(spectral_rolloff), np.min(spectral_rolloff), np.std(spectral_rolloff)])

    return features

def make_speech_features():

    df = pd.read_csv('iemocap_metadata.csv')
    df.loc[(df['emotion'] == 'exc'), 'emotion'] = 'hap'
    df.drop(df.loc[(df['emotion'] == 'xxx') | (df['emotion'] == 'dis') | (df['emotion'] == 'oth') | (df['emotion'] == 'fea') | (df['emotion'] == 'sur')].index, inplace = True)

    file_list = df['path'].tolist()
    emotions = df['emotion'].tolist()

    dataframe = []
    for index, file_name in enumerate(file_list):

        speech_features = _extract_features('data/IEMOCAP_dataset/' + file_name)
        speech_features.append(emotions[index])
        speech_features.append(str(os.path.basename(file_name))[:-4] )
        speech_features = tuple(speech_features)

        dataframe.append(speech_features)

        print('On file number ', index + 1, '/', len(file_list))
    
    columns = ['mfccs_mean', 'mfccs_median', 'mfccs_max', 'mfccs_min', 'mfccs_std', 
                'mel_mean', 'mel_median', 'mel_max', 'mel_min', 'mel_std',
                'pitch_mean', 'pitch_median', 'pitch_max', 'pitch_min', 'pitch_std',
                'rms_mean', 'rms_median', 'rms_max', 'rms_min', 'rms_std',
                'zcr_mean', 'zcr_median', 'zcr_max', 'zcr_min', 'zcr_std',
                'chroma_mean', 'chroma_median', 'chroma_max', 'chroma_min', 'chroma_std', 
                'spectral_flux_mean', 'spectral_flux_median', 'spectral_flux_max', 'spectral_flux_min', 'spectral_flux_std', 
                'spectral_rolloff_mean', 'spectral_rolloff_median', 'spectral_rolloff_max', 'spectral_rolloff_min', 'spectral_rolloff_std', 
                'emotion', 'file_name']
    
    df = pd.DataFrame(dataframe, columns=columns)
    df.to_csv('file.csv', index=False)

if __name__ == '__main__':
    make_speech_features()