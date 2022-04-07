from scipy.io import wavfile
import noisereduce as nr
import pandas as pd
import os

def main():

    df = pd.read_csv('iemocap_metadata.csv')
    file_list = df['path'].tolist()

    for index, file_name in enumerate(file_list):
        print('On file number', index + 1, '/', len(file_list))
        denoise(file_name)


def denoise(file_name):
    rate, data = wavfile.read('data/IEMOCAP_dataset/' + file_name)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    path_to_new_file = 'data/IEMOCAP_dataset_denoised/' + file_name
    if not os.path.exists(os.path.dirname(path_to_new_file)):
        os.makedirs(os.path.dirname(path_to_new_file))

    wavfile.write(path_to_new_file, rate, reduced_noise)

if __name__ == '__main__':
    main()