from sklearn.metrics import classification_report
from speech_models import speech_logistic_regression
from process_dataset import speech_features, text_features
from ensemble import SpeechTextEnsemble, StackEnsemble, VoteEnsemble, BlendEnsemble
import numpy as np
import pandas as pd
import pickle

def main():

    meta_cls = speech_logistic_regression.get_logistic_regression()

    # speech_model = BlendEnsemble(meta_cls=meta_cls, data_type='speech')
    # speech_model = StackEnsemble(meta_cls=meta_cls, data_type='speech')
    # speech_model = VoteEnsemble(type='soft', data_type='speech')
    # speech_model = VoteEnsemble(type='hard', data_type='speech')

    # text_model = BlendEnsemble(meta_cls=meta_cls, data_type='text')
    # text_model = StackEnsemble(meta_cls=meta_cls, data_type='text')
    # text_model = VoteEnsemble(type='soft', data_type='text')
    # text_model = VoteEnsemble(type='hard', data_type='text')

    # with open('trained_models/stack_speech.pkl', 'rb') as f:
    #     speech_model = pickle.load(f)

    # To check accuracy
    # check_accuracy(text_model, data_type='text', save_name='stack_text.pkl')

    # To cross validate
    # cross_validate(text_model, data_type='text', cv=5)

    combined_model = SpeechTextEnsemble(fit_bases=False)
    x_train_s, x_test_s, y_train_s, y_test_s = speech_features.get_train_test()
    x_train_t, x_test_t, y_train_t, y_test_t = text_features.get_train_test()
    result = combined_model.predict(x_test_s, x_test_t)
    print(classification_report(y_test_s, result))


    # result = []

    # for proba_speech, proba_text in zip(probas_speech, probas_text):
    #     avg_proba = (proba_speech + proba_text) / 2
    #     max_index_speech, max_index_text = np.argmax([proba_speech, proba_text], axis=1)

    #     if max_index_speech != max_index_text:
    #         # speech model -> good at sad(3), neutral(2)
    #         # text model -> good at ang(0), hap(1)

    #         if max_index_speech in [2, 3] and max_index_text in [2, 3]:
    #             # only speech giving max proba at the emo they good at
    #             max_index = max_index_speech
            
    #         elif max_index_text in [0, 1] and max_index_speech in [0, 1]:
    #             # only text giving max proba at the emo they good at
    #             max_index = max_index_text

    #         else:
    #             max_index = np.argmax(avg_proba)

    #     else:

    #         max_index = max_index_speech

    #     result.append(emotions[max_index])




def check_accuracy(model, data_type, save_name=None):
    if data_type == 'text':
        x_train, x_test, y_train, y_test = text_features.get_train_test()
    else:
        x_train, x_test, y_train, y_test = speech_features.get_train_test() 
    
    model.fit(x_train, y_train)

    if save_name != None:
        model.save(save_name)

    results = model.predict(x_test)
    print(classification_report(y_test, results))

def cross_validate(model, data_type, cv, save_name=None):
    if data_type == 'text':
        x, y = text_features.get_data()
    else:
        x, y = speech_features.get_data()

    if save_name != None:
        model.save(save_name)

    scoring = {'accuracy': 'accuracy',
           'f1_macro': 'f1_macro',
           'precision_macro': 'precision_macro',
           'recall_macro' : 'recall_macro'}

    scores = model.cross_validate(x, y, cv=cv, scoring=scoring)
    print('Accuracy: ', np.mean(scores['test_accuracy']))
    print('F1 Macro: ', np.mean(scores['test_f1_macro']))
    print('Precision Macro: ', np.mean(scores['test_precision_macro']))
    print('Recall Macro: ', np.mean(scores['test_recall_macro']))

if __name__ == '__main__':
    main()