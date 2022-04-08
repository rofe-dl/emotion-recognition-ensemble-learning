from sklearn.metrics import classification_report
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm
from process_dataset import speech_features, text_features
from ensemble import StackEnsemble, VoteEnsemble, BlendEnsemble
import numpy as np
import pandas as pd
import pickle

def main():

    meta_cls = speech_logistic_regression.get_logistic_regression()

    # speech_model = BlendEnsemble(meta_cls=meta_cls, data_type='speech')
    speech_model = StackEnsemble(meta_cls=meta_cls, data_type='speech')
    # speech_model = VoteEnsemble(type='soft', data_type='speech')
    # speech_model = VoteEnsemble(type='hard', data_type='speech')

    # text_model = BlendEnsemble(meta_cls=meta_cls, data_type='text')
    text_model = StackEnsemble(meta_cls=meta_cls, data_type='text')
    # text_model = VoteEnsemble(type='soft', data_type='text')
    # text_model = VoteEnsemble(type='hard', data_type='text')

    # To check accuracy
    # check_accuracy_speech(model)
    # check_accuracy_text(model)
    # To cross validate
    # cross_validate_speech(model)
    # cross_validate_text(model)

    x_train1, x_test1, y_train1, y_test1 = speech_features.get_train_test()
    x_train2, x_test2, y_train2, y_test2 = text_features.get_train_test()

    # speech_model.fit(x_train1, y_train1)
    # with open('trained_models/speech_stack.pkl', 'wb') as f:
    #     pickle.dump(speech_model, f)

    with open('trained_models/speech_stack.pkl', 'rb') as f:
        speech_model = pickle.load(f)
    probas_speech = speech_model.predict(x_test1, proba=True)

    # text_model.fit(x_train2, y_train2)
    # with open('trained_models/text_stack.pkl', 'wb') as f:
    #     pickle.dump(text_model, f)

    with open('trained_models/text_stack.pkl', 'rb') as f:
        text_model = pickle.load(f)
    probas_text = text_model.predict(x_test2, proba=True)

    avg_probas = (probas_speech + probas_text) / 2
    emotions = ['ang', 'hap', 'neu', 'sad']
    #             0      1      2      3

    result = []

    for proba_speech, proba_text in zip(probas_speech, probas_text):
        avg_proba = (proba_speech + proba_text) / 2
        max_index_speech, max_index_text = np.argmax([proba_speech, proba_text], axis=1)

        if max_index_speech != max_index_text:
            # speech model -> good at sad(3), neutral(2)
            # text model -> good at ang(0), hap(1)

            if (max_index_speech in [2, 3] and max_index_text in [0, 1]):
                # both giving max proba at the emo they're good at, then average their probas
                max_index = np.argmax(avg_proba)

            elif max_index_speech in [2, 3] and max_index_text in [2, 3]:
                # only speech giving max proba at the emo they good at
                max_index = max_index_speech
            
            elif max_index_text in [0, 1] and max_index_speech in [0, 1]:
                # only text giving max proba at the emo they good at
                max_index = max_index_text
            else:
                max_index = np.argmax(avg_proba)

        else:
            max_index = max_index_speech

        result.append(emotions[max_index])

    # max_indices = np.argmax(avg_probas, axis=1)
    # for index in max_indices:
    #     result.append(emotions[index])

    print(classification_report(y_test1, result))

    # df1 = pd.DataFrame(speech_model.predict(x_test1, proba=True), columns=speech_model.inner.classes_)
    # print(df1)

    # df2 = pd.DataFrame(text_model.predict(x_test2, proba=True), columns=text_model.inner.classes_)
    # print(df2)



def check_accuracy_speech(model):
    x_train, x_test, y_train, y_test = speech_features.get_train_test()
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    print(classification_report(y_test, results))

def cross_validate_speech(model):
    x, y = speech_features.get_data()
    scores = model.cross_validate(x, y, cv=5, scoring='f1_macro')
    print(scores.mean())

def check_accuracy_text(model):
    x_train, x_test, y_train, y_test = text_features.get_train_test()
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    print(classification_report(y_test, results))

def cross_validate_text(model):
    x, y = text_features.get_data()
    scores = model.cross_validate(x, y, cv=5, scoring='f1_macro')
    print(scores.mean())

if __name__ == '__main__':
    main()