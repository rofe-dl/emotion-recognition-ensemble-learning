import pickle
with open('data/speech_features.pkl', 'rb') as f:
    features = pickle.load(f)

print(len(features[1]))