from process_dataset.speech_features import get_speech_features
import pickle
import numpy as np
import pandas as pd

# data = get_speech_features()
# new_pkl = list()

# for i in range(len(data[0])):
#     arr = [data[0][i], data[1][i], data[2][i], data[3][i], data[4][i], data[5][i]]
#     new_pkl.append(np.array(arr))

# with open('speech_features_separated_iemocap.pkl', 'wb') as f:
#     pickle.dump(new_pkl, f)