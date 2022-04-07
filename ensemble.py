from sklearn.model_selection import cross_val_score
import numpy as np
from speech_models import speech_logistic_regression, speech_mlp, speech_naive_bayes
from speech_models import speech_random_forest, speech_svm, speech_xgboost
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore') 

def get_speech_models():

    svm_ = speech_svm.get_svm()
    rfc_ = speech_random_forest.get_random_forest()
    mnb_ = speech_naive_bayes.get_naive_bayes()
    lr_ = speech_logistic_regression.get_logistic_regression()
    mlp_ = speech_mlp.get_mlp()
    xgb_ = speech_xgboost.get_xgb()

    # TODO xgboost and lstm 

    models = [('Support Vector Machine', svm_), ('Random Forest Classifier', rfc_), ('Multinomial Naive Bayes', mnb_),
                ('Logistic Regression', lr_), ('MLP Classifier', mlp_), ('XGBoost', xgb_)]
    
    return models

def get_text_models():
    
    # TODO xgboost and lstm 
    pass


class Ensemble:

    def __init__(self, models):
        self.models = models

    def save(self):
        pass

class StackEnsemble(Ensemble):

    def __init__(self, meta_cls, data_type='speech'):
        if data_type == 'speech':
            super().__init__(models=get_speech_models())
        else:
            super().__init__(models=get_text_models())

        self.data_type = data_type
        self.meta_cls = meta_cls
        
    def fit(self, x_train, y_train):
        self.init_stacker()
        self.stacker.fit(x_train, y_train)
    
    def init_stacker(self):
        self.stacker = StackingClassifier(
            estimators=self.models,
            final_estimator=self.meta_cls,
            stack_method='predict_proba',
            cv=5, verbose=1,
            n_jobs=-1)

    def predict(self, x_test):
        return self.stacker.predict(x_test)
    
    def cross_validate(self, x, y, cv, scoring):
        self.init_stacker()
        return cross_val_score(self.stacker, x, y, cv=cv, scoring=scoring, n_jobs=-1)

class VoteEnsemble(Ensemble):
    
    def __init__(self, type, data_type='speech'):
        if data_type == 'speech':
            super().__init__(models=get_speech_models())
        else:
            super().__init__(models=get_text_models())
            
        self.data_type = data_type
        self.type = type
    
    def fit(self, x_train, y_train):
        self.init_voter()
        self.voter.fit(x_train, y_train)
    
    def init_voter(self):
        self.voter = VotingClassifier(
            estimators=self.models, 
            voting=self.type,
            verbose=True, 
            n_jobs=-1)

    def predict(self, x_test):
        return self.voter.predict(x_test)

    def cross_validate(self, x, y, cv, scoring):
        self.init_voter()
        return cross_val_score(self.voter, x, y, cv=cv, scoring=scoring, n_jobs=-1)

class BlendEnsemble(Ensemble):

    def __init__(self, meta_cls, data_type='speech'):
        if data_type == 'speech':
            super().__init__(models=get_speech_models())
        else:
            super().__init__(models=get_text_models())
            
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
        return cross_val_score(self, x, y, cv=cv, scoring=scoring, n_jobs=-1, error_score='raise')


