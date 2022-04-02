import ensemble
import pickle 

# from process_dataset.speech_features import get_speech_features

# data = get_speech_features()
# new_x, new_y = [], []
# for x, y in zip(data[0], data[1]):

#     if y == 'fea'or y == 'sur':
#         continue
    
#     new_x.append(x)
#     new_y.append(y)

# features = (new_x, new_y)
# with open('speech_features_hstacked_iemocap_cleaned_2.pkl', 'wb') as f:
#     pickle.dump(features, f)

def main():
    ensemble.stacking_ensemble()
    # ensemble.voting_ensemble()
    
if __name__ == '__main__':
    main()