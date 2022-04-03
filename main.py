from sklearn.metrics import classification_report
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm
from process_dataset.speech_features import get_data, get_train_test
from ensemble import StackEnsemble, VoteEnsemble, BlendEnsemble

def main():

    # print('Voting Ensemble')
    # voter = VoteEnsemble(type='soft')
    # # To check accuracy
    # x_train, x_test, y_train, y_test = get_train_test()
    # voter.fit(x_train, y_train)
    # results = voter.predict(x_test)
    # print(classification_report(y_test, results))
    # # To cross validate
    # x, y = get_data()
    # scores = voter.cross_validate(x, y, cv=5, scoring='f1_macro')
    # print(scores.mean())


    # print('Stacking Ensemble')
    # meta_cls = speech_logistic_regression.get_logistic_regression()
    # stack = StackEnsemble(meta_cls=meta_cls)
    # # To check accuracy
    # x_train, x_test, y_train, y_test = get_train_test()
    # stack.fit(x_train, y_train)
    # results = stack.predict(x_test)
    # print(classification_report(y_test, results))
    # To cross validate
    # x, y = get_data()
    # scores = stack.cross_validate(x, y, cv=5, scoring='f1_macro')
    # print(scores.mean())

    print('Blending Ensemble')
    meta_cls = speech_logistic_regression.get_logistic_regression()
    blend = BlendEnsemble(meta_cls=meta_cls, model_type='speech')

    # # To check accuracy
    # x_train, x_test, y_train, y_test = get_train_test()
    # blend.fit(x_train, y_train)
    # results = blend.predict(x_test)
    # print(classification_report(y_test, results))
    # To cross validate
    x, y = get_data()
    scores = blend.cross_validate(x, y, cv=5, scoring='f1_macro')
    print(scores.mean())

    
if __name__ == '__main__':
    main()