from sklearn.metrics import classification_report
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm
from process_dataset import speech_features, text_features
from ensemble import StackEnsemble, VoteEnsemble, BlendEnsemble
import numpy as np
import pandas as pd

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

    speech_model.fit(x_train1, y_train1)
    speech_result = speech_model.predict(x_test1, proba=True)

    text_model.fit(x_train2, y_train2)
    text_result = text_model.predict(x_test2, proba=True)

    probas = (speech_result + text_result) / 2

    emotions = ['ang', 'hap', 'neu', 'sad']
    result = []

    for proba in probas:
        proba = list(proba)
        max_proba = max(proba)
        max_index = proba.index(max_proba)
        result.append(emotions[max_index])

    result = np.array(result)
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