import numpy as np
import pickle
import os

from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes, speech_random_forest, speech_svm, speech_xgboost
from text_models import text_logistic_regression, text_mlp, text_naive_bayes, text_random_forest, text_svm, text_xgboost

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

import warnings
warnings.filterwarnings('ignore') 

def get_speech_models():

    models = list()

    models.append(('Support Vector Machine', speech_svm.get_svm()))
    models.append(('Random Forest Classifier', speech_random_forest.get_random_forest()))
    models.append(('Multinomial Naive Bayes', speech_naive_bayes.get_naive_bayes()))
    models.append(('Logistic Regression', speech_logistic_regression.get_logistic_regression()))
    models.append(('MLP Classifier', speech_mlp.get_mlp()))
    models.append(('XGBoost', speech_xgboost.get_xgb()))

    # TODO lstm 

    return models

def get_text_models():
    
    models = list()

    models.append(('Support Vector Machine', text_svm.get_svm()))
    models.append(('Random Forest Classifier', text_random_forest.get_random_forest()))
    models.append(('Multinomial Naive Bayes', text_naive_bayes.get_naive_bayes()))
    models.append(('Logistic Regression', text_logistic_regression.get_logistic_regression()))
    models.append(('MLP Classifier', text_mlp.get_mlp()))
    models.append(('XGBoost', text_xgboost.get_xgb()))

    # TODO lstm 

    return models

class Ensemble:

    def __init__(self, data_type):
        if data_type == 'speech':
            self.models = get_speech_models()
        elif data_type == 'text':
            self.models = get_text_models()
        else:
            self.models = None

    def save(self, file_name):
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')

        with open(f"trained_models/{file_name}", 'wb') as f:
            pickle.dump(self, f)

class StackEnsemble(Ensemble):

    def __init__(self, meta_cls, data_type='speech'):
        super().__init__(data_type=data_type)

        self.data_type = data_type
        self.meta_cls = meta_cls
        
    def fit(self, x_train, y_train):
        self.init_inner()
        self.inner.fit(x_train, y_train)
    
    def init_inner(self):
        self.inner = StackingClassifier(
            estimators=self.models,
            final_estimator=self.meta_cls,
            stack_method='predict_proba',
            cv=5, verbose=1,
            n_jobs=-1)

    def predict(self, x_test, proba=False):
        return self.inner.predict_proba(x_test) if proba else self.inner.predict(x_test)
    
    def cross_validate(self, x, y, cv, scoring):
        self.init_inner()
        return cross_validate(self.inner, x, y, cv=cv, scoring=scoring, n_jobs=-1)

class VoteEnsemble(Ensemble):
    
    def __init__(self, type, data_type='speech'):
        super().__init__(data_type=data_type)
            
        self.data_type = data_type
        self.type = type
    
    def fit(self, x_train, y_train):
        self.init_inner()
        self.inner.fit(x_train, y_train)
    
    def init_inner(self):
        self.inner = VotingClassifier(
            estimators=self.models, 
            voting=self.type,
            verbose=True, 
            n_jobs=-1)

    def predict(self, x_test, proba=False):
        return self.inner.predict_proba(x_test) if proba else self.inner.predict(x_test)

    def cross_validate(self, x, y, cv, scoring):
        self.init_inner()
        return cross_validate(self.inner, x, y, cv=cv, scoring=scoring, n_jobs=-1)

class BlendEnsemble(Ensemble):

    def __init__(self, meta_cls, data_type='speech'):
        super().__init__(data_type=data_type)
            
        self.data_type = data_type
        self.meta_cls = meta_cls

    def fit(self, x_train, y_train, val_size=0.25):

        # creates validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42) # 0.25 x 0.8 = 0.2
        meta_x = list()

        for model_name, model in self.models:
            model.fit(x_train, y_train)
            y_pred = model.predict_proba(x_val)
            meta_x.append(y_pred)

        meta_x = np.hstack(meta_x)
        self.meta_cls.fit(meta_x, y_val)

    def predict(self, x_test):
        meta_x = list()

        for model_name, model in self.models:
            y_pred = model.predict_proba(x_test)
            meta_x.append(y_pred)

        meta_x = np.hstack(meta_x)
        return self.meta_cls.predict(meta_x)
    
    # Needed to call cross_val_score on custom model
    def get_params(self, deep=True):
        return {"meta_cls": self.meta_cls, "data_type": self.data_type}

    # Needed to call cross_val_score on custom model
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def cross_validate(self, x, y, cv, scoring):
        return cross_validate(self, x, y, cv=cv, scoring=scoring, n_jobs=-1)

# Used to ensemble the speech and text ensembled models by soft voting
class SpeechTextEnsemble(Ensemble):
    
    def __init__(self, speech_model=None, text_model=None, fit_bases=True):
        super().__init__(data_type=None)

        if fit_bases and (speech_model == None or text_model == None):
            raise ValueError("speech_model or text_model can't be None when fit_bases is True")

        if fit_bases:
            self._speech_model = speech_model
            self._text_model = text_model
        else:
            with open('trained_models/stack_speech.pkl', 'rb') as f:
                self._speech_model = pickle.load(f)
            with open('trained_models/stack_text.pkl', 'rb') as f:
                self._text_model = pickle.load(f)

        self._fit_bases = fit_bases

    def fit(self, x_train_speech, x_train_text, y_train):

        if self._fit_bases:
            self._speech_model.fit(x_train_speech, y_train)
            self._text_model.fit(x_train_text, y_train)


    def predict(self, x_test_speech, x_test_text, proba=False):

        probas_speech = self._speech_model.predict(x_test_speech, proba=True)
        probas_text = self._text_model.predict(x_test_text, proba=True)

        avg_probas = (probas_speech + probas_text) / 2

        if proba:
            return avg_probas

        emotions = ['ang', 'hap', 'neu', 'sad']
        #             0      1      2      3

        result = []

        max_indices = np.argmax(avg_probas, axis=1)
        for index in max_indices:
            result.append(emotions[index])

        return result

    # Needed to call cross_val_score on custom model
    def get_params(self, deep=True):
        return {"_speech_model": self._speech_model, 
                "_text_model": self._text_model,
                "fit_bases": True}

    # Needed to call cross_val_score on custom model
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        


    