from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import resample, shuffle
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from process_dataset.speech_features import get_speech_features

import warnings
warnings.filterwarnings('ignore') 

def get_models():

    svm_ = SVC(kernel='linear', probability=True, random_state=42)
    rfc_ = RandomForestClassifier(random_state=42, n_jobs=-1)
    mnb_ = MultinomialNB(alpha=0.01)
    lr_ = LogisticRegression(solver='newton-cg', random_state=42, n_jobs=-1)
    mlp_ = MLPClassifier(random_state=42, max_iter=5000)

    models = [('Support Vector Machine', svm_), ('Random Forest Classifier', rfc_), ('Multinomial Naive Bayes', mnb_),
                ('Logistic Regression', lr_), ('MLP Classifier', mlp_)]
    
    return models

def get_train_test():
    # Get preprocessed dataset and scale them
    data = get_speech_features()
    x = np.array(data[0])
    y = np.array(data[1])

    x = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    return x_train, x_test, y_train, y_test

def test_models(models):
    # For classification report
    x_train, x_test, y_train, y_test = get_train_test()

    # To cross validate
    # data = get_speech_features()
    # x = np.array(data[0])
    # y = np.array(data[1])
    # x = MinMaxScaler().fit_transform(x)

    for model_name, model in models:
        print("================================")
        print(model_name)
        
        # To cross validate
        # scores = cross_val_score(model, x, y, cv=5, scoring='f1_macro', n_jobs=-1)
        # print(scores.mean())

        # For classification report
        model.fit(x_train, y_train)
        results = model.predict(x_test)
        print('Accuracy: ', accuracy_score(y_test, results))
        print(classification_report(y_test, results))

def stacking_ensemble():
    models = get_models()

    meta_cls = LogisticRegression(random_state=42, solver='newton-cg', n_jobs=-1)
    stack = sklearn.ensemble.StackingClassifier(estimators=list(models), final_estimator=meta_cls, stack_method='predict_proba', cv=5, verbose=1, n_jobs=-1)

    models.append(('STACKED ENSEMBLE', stack))

    test_models(models)

def voting_ensemble():
    pass