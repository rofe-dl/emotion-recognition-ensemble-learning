from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

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

def get_data():
    # Get preprocessed dataset and scale them
    data = get_speech_features()
    x = np.array(data[0])
    y = np.array(data[1])

    x = MinMaxScaler().fit_transform(x)

    return x, y

def get_train_test():
    # Get preprocessed dataset and scale them
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    return x_train, x_test, y_train, y_test

def get_train_val_test():
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    return x_train, x_val, x_test, y_train, y_val, y_test

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
    stack = StackingClassifier(estimators=list(models), final_estimator=meta_cls, stack_method='predict_proba', cv=5, verbose=1, n_jobs=-1)

    models.append(('STACK ENSEMBLE', stack))

    test_models(models)

def hard_voting_ensemble():
    models = get_models()
    voter = VotingClassifier(estimators=list(models), voting='hard', verbose=True, n_jobs=-1)

    models.append(('HARD VOTE ENSEMBLE', voter))
    
    test_models((models))

def soft_voting_ensemble():
    models = get_models()
    voter = VotingClassifier(estimators=list(models), voting='soft', verbose=True, n_jobs=-1)

    models.append(('SOFT VOTE ENSEMBLE', voter))
    
    test_models((models))

class BlendEnsemble:

    def __init__(self, meta_cls):
        self.models = []
        self.meta_cls = meta_cls
    
    def add_models(self, models):
        self.models.extend(models)

    def fit(self, x_train, x_val, y_train, y_val):
        meta_x = list()

        for model_name, model in self.models:
            model.fit(x_train, y_train)
            y_pred = model.predict_proba(x_val)
            print(y_pred)
            # y_pred = y_pred.reshape(len(y_pred), 1)

            meta_x.append(y_pred)
        
        meta_x = np.hstack(meta_x)
        self.meta_cls.fit(meta_x, y_val)

    def predict(self, x_test):
        meta_x = list()

        for model_name, model in self.models:
            y_pred = model.predict_proba(x_test)
            # y_pred = y_pred.reshape(len(y_pred), 1)

            meta_x.append(y_pred)

        meta_x = np.hstack(meta_x)
        return self.meta_cls.predict(meta_x)

def blend_ensemble():
    models = get_models()
    meta_cls = LogisticRegression(random_state=42, solver='newton-cg', n_jobs=-1)

    blender = BlendEnsemble(meta_cls=meta_cls)
    blender.add_models(list(models))

    x_train, x_val, x_test, y_train, y_val, y_test = get_train_val_test()
    models.append(('BLEND ENSEMBLE', blender))
    for model_name, model in models:
        print("================================")
        print(model_name)
        
        # To cross validate
        # scores = cross_val_score(model, x, y, cv=5, scoring='f1_macro', n_jobs=-1)
        # print(scores.mean())

        # For classification report
        if model_name == 'BLEND ENSEMBLE':
            model.fit(x_train, x_val, y_train, y_val)
        else:
            model.fit(x_train, y_train)

        results = model.predict(x_test)
        print('Accuracy: ', accuracy_score(y_test, results))
        print(classification_report(y_test, results))


