from sklearn.metrics import classification_report
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm
from process_dataset import speech_features, text_features
from ensemble import StackEnsemble, VoteEnsemble, BlendEnsemble
import numpy as np

def main():

    meta_cls = speech_logistic_regression.get_logistic_regression()

    # model = BlendEnsemble(meta_cls=meta_cls, data_type='speech')
    model = StackEnsemble(meta_cls=meta_cls, data_type='speech')
    # model = VoteEnsemble(type='soft', data_type='speech')
    # model = VoteEnsemble(type='hard', data_type='speech')

    # model = BlendEnsemble(meta_cls=meta_cls, data_type='text')
    # model = StackEnsemble(meta_cls=meta_cls, data_type='text')
    # model = VoteEnsemble(type='soft', data_type='text')
    # model = VoteEnsemble(type='hard', data_type='text')

    # To check accuracy
    check_accuracy_speech(model)
    # check_accuracy_text(model)
    # To cross validate
    # cross_validate_speech(model)
    # cross_validate_text(model)

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