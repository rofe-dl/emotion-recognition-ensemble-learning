from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm
from process_dataset.speech_features import get_data, get_train_test, get_train_val_test
from ensemble import StackEnsemble, VoteEnsemble, BlendEnsemble

def get_speech_models():

    svm_ = speech_svm.get_svm()
    rfc_ = speech_random_forest.get_random_forest()
    mnb_ = speech_naive_bayes.get_naive_bayes()
    lr_ = speech_logistic_regression.get_logistic_regression()
    mlp_ = speech_mlp.get_mlp()

    models = [('Support Vector Machine', svm_), ('Random Forest Classifier', rfc_), ('Multinomial Naive Bayes', mnb_),
                ('Logistic Regression', lr_), ('MLP Classifier', mlp_)]
    
    return models

def main():

    # print('Voting Ensemble')
    # voter = VoteEnsemble(type='soft')
    # voter.add_models(get_speech_models())
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
    # stack = StackEnsemble(meta_cls=speech_logistic_regression.get_logistic_regression())
    # stack.add_models(get_speech_models())
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
    blend = BlendEnsemble(meta_cls=speech_logistic_regression.get_logistic_regression())
    blend.add_models(get_speech_models())
    # # To check accuracy
    # x_train, x_val, x_test, y_train, y_val, y_test = get_train_val_test()
    # blend.fit(x_train, x_val, y_train, y_val)
    # results = blend.predict(x_test)
    # print(classification_report(y_test, results))

    
if __name__ == '__main__':
    main()