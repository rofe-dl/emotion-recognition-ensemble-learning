from sklearn.model_selection import cross_val_score
import numpy as np


from sklearn.ensemble import VotingClassifier, StackingClassifier

from process_dataset.speech_features import get_train_test, get_train_val_test

import warnings
warnings.filterwarnings('ignore') 

class Ensemble:

    def __init__(self):
        self.models = []

    def add_models(self, models):
        self.models.extend(models)

class StackEnsemble(Ensemble):

    def __init__(self, meta_cls):
        super().__init__()
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
    
    def __init__(self, type):
        super().__init__()
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

    def save(self):
        pass

class BlendEnsemble(Ensemble):

    def __init__(self, meta_cls):
        super().__init__()
        self.meta_cls = meta_cls

    def fit(self, x_train, x_val, y_train, y_val):
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


