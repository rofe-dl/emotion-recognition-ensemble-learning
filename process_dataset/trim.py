import librosa
import soundfile as sf
import pandas as pd
import os

def main():

    df = pd.read_csv('iemocap_metadata.csv')
    file_list = df['path'].tolist()

    for index, file_name in enumerate(file_list):
        print('On file number', index + 1, '/', len(file_list))
        trim(file_name)

def trim(file_name):

    old_file_path = 'data/IEMOCAP_dataset_denoised/' + file_name
    with sf.SoundFile(old_file_path) as sound_file:
        sample_rate = sound_file.samplerate

    audio, sr = librosa.load(old_file_path, sr=sample_rate, mono=True)
    clip = librosa.effects.trim(audio, top_db=30)

    path_to_new_file = 'data/IEMOCAP_dataset_denoised_trimmed/' + file_name
    if not os.path.exists(os.path.dirname(path_to_new_file)):
        os.makedirs(os.path.dirname(path_to_new_file))

    # sf.write(path_to_new_file, wav_data, sr)
    sf.write(path_to_new_file, clip[0], sr)

if __name__ == '__main__':
    main()