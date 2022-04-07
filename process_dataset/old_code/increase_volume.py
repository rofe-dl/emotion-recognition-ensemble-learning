import librosa
import soundfile as sf
import pandas as pd
import os
from pydub import AudioSegment

def main():

    df = pd.read_csv('iemocap_metadata.csv')
    file_list = df['path'].tolist()

    for index, file_name in enumerate(file_list):
        print('On file number', index + 1, '/', len(file_list))
        increase(file_name)

def increase(file_name):

    old_file_path = 'data/IEMOCAP_dataset_denoised_trimmed/' + file_name
    audio = AudioSegment.from_wav(old_file_path)

    path_to_new_file = 'data/IEMOCAP_dataset_denoised_trimmed_increased/' + file_name
    if not os.path.exists(os.path.dirname(path_to_new_file)):
        os.makedirs(os.path.dirname(path_to_new_file))

    audio = audio + 17 # increase volume by 50dB
    audio.export(path_to_new_file, format='wav')

if __name__ == '__main__':
    main()