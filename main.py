from sklearn.metrics import classification_report
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm
from process_dataset.speech_features import get_data, get_train_test
from ensemble import StackEnsemble, VoteEnsemble, BlendEnsemble
import numpy as np

def main():

    meta_cls = speech_logistic_regression.get_logistic_regression()

    # model = BlendEnsemble(meta_cls=meta_cls, data_type='speech')
    # model = StackEnsemble(meta_cls=meta_cls, data_type='speech')
    # model = VoteEnsemble(type='soft', data_type='speech')
    model = VoteEnsemble(type='hard', data_type='speech')

    # To check accuracy
    check_accuracy(model)
    # To cross validate
    # cross_validate(model)

def check_accuracy(model):
    x_train, x_test, y_train, y_test = get_train_test()
    model.fit(x_train, y_train)
    results = model.predict(x_test)
    print(classification_report(y_test, results))

def cross_validate(model):
    x, y = get_data()
    scores = model.cross_validate(x, y, cv=5, scoring='f1_macro')
    print(scores.mean())

if __name__ == '__main__':
    main()