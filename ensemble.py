import numpy as np
import copy
import os
import pickle

from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings('ignore')

class Ensemble():

    def save(self, file_name):
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')

        with open(f"trained_models/{file_name}", 'wb') as f:
            pickle.dump(self, f)

    def cross_validate(self, x_speech, x_text, y, cv):

        # wrong impementation, does overfitting because of refit

        if len(x_speech) < cv:
            raise ValueError("Dataset is too small for k")

        accuracies, f1s, precisions, recalls = (list() for i in range(4))
        x_speech_split = np.array_split(x_speech, cv)
        x_text_split = np.array_split(x_text, cv)
        y_split = np.array_split(y, cv)

        for i in range(cv):
            x_speech_test = x_speech_split[i]
            x_text_test = x_text_split[i]
            y_test = y_split[i]

            x_speech_train = np.concatenate([split for index, split in enumerate(x_speech_split) if index != i])
            x_text_train = np.concatenate([split for index, split in enumerate(x_text_split) if index != i])
            y_train = np.concatenate([split for index, split in enumerate(y_split) if index != i])

            scaler = MinMaxScaler()
            x_speech_train = scaler.fit_transform(x_speech_train)
            x_speech_test = scaler.transform(x_speech_test)

            vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2))
            x_text_train = vectorizer.fit_transform(x_text_train)
            x_text_test = vectorizer.transform(x_text_test)

            self.fit(x_speech_train, x_text_train, y_train)
            result = self.predict(x_speech_test, x_text_test)

            accuracies.append(accuracy_score(y_test, result))
            f1s.append(f1_score(y_test, result, average='macro'))
            precisions.append(precision_score(y_test, result, average='macro'))
            recalls.append(recall_score(y_test, result, average='macro'))


        return {
            'test_accuracy': accuracies,
            'test_f1_macro': f1s,
            'test_precision_macro': precisions,
            'test_recall_macro': recalls
        }

class VoteEnsemble(Ensemble):

    def __init__(self, speech_models, text_models, type):
        self.speech_models = speech_models
        self.text_models = text_models
        self.type = type
        self.voter_speech = VotingClassifier(
            estimators=self.speech_models, 
            voting=self.type,
            verbose=True,
            n_jobs=-1
        )
        self.voter_text = VotingClassifier(
            estimators=self.text_models, 
            voting=self.type,
            verbose=True,
            n_jobs=-1
        )

    def fit(self, x_speech, x_text, y):

        self.voter_speech.fit(x_speech, y)
        self.voter_text.fit(x_text, y)

    def predict(self, x_speech, x_text, proba=False):

        probas_speech = self.voter_speech.predict_proba(x_speech)
        probas_text = self.voter_text.predict_proba(x_text)

        avg_probas = (probas_speech + probas_text) / 2

        if proba:
            return avg_probas

        emotions = ['ang', 'hap', 'neu', 'sad']
        result = []

        max_indices = np.argmax(avg_probas, axis=1)
        for index in max_indices:
            result.append(emotions[index])

        return result


class BlendEnsemble(Ensemble):

    def __init__(self, speech_models, text_models, meta_cls):
        self.speech_models = speech_models
        self.text_models = text_models
        self.meta_cls = meta_cls
    
    def fit(self, x_speech, x_text, y, val_size=0.25):
        meta_x = None

        # creates validation sets
        x_speech_train, x_speech_val, y_train, y_val = train_test_split(x_speech, y, test_size=val_size, random_state=42) # 0.25 x 0.8 = 0.2
        x_text_train, x_text_val, y_train, y_val = train_test_split(x_text, y, test_size=val_size, random_state=42) # 0.25 x 0.8 = 0.2

        for name, model in self.speech_models:
            print(f"Training {name} (Speech) ...")
            model.fit(x_speech_train, y_train)
            probas = model.predict_proba(x_speech_val)

            if meta_x is None:
                meta_x = probas
            else:
                meta_x = np.column_stack((meta_x, probas))

        for name, model in self.text_models:
            print(f"Training {name} (Text) ...")
            model.fit(x_text_train, y_train)
            probas = model.predict_proba(x_text_val)

            if meta_x is None:
                meta_x = probas
            else:
                meta_x = np.column_stack((meta_x, probas))

        print("Training Meta Classifier ...")
        self.meta_cls.fit(meta_x, y_val)

    def predict(self, x_speech, x_text, proba=False):
        meta_x = list()

        for name, model in self.speech_models:
            probas = model.predict_proba(x_speech)
            meta_x.append(probas)
        
        for name, model in self.text_models:
            probas = model.predict_proba(x_text)
            meta_x.append(probas)

        meta_x = np.hstack(meta_x)

        return self.meta_cls.predict(meta_x) if not proba else self.meta_cls.predict_proba(meta_x)


class StackEnsemble(Ensemble):
    
    def __init__ (self, speech_models, text_models, meta_cls, cv, n_jobs):
        self.speech_models = speech_models
        self.text_models = text_models
        self.meta_cls = meta_cls
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, x_speech, x_text, y):

        meta_x = None

        speech_models = copy.deepcopy(self.speech_models)
        text_models = copy.deepcopy(self.text_models)

        for name, model in speech_models:
            print(f"Training {name} (Speech) ...")
            probas = cross_val_predict(
                model, x_speech, y, cv=self.cv, 
                n_jobs=self.n_jobs, method='predict_proba'
            )

            if meta_x is None:
                meta_x = probas
            else:
                meta_x = np.column_stack((meta_x, probas)) # hstack also works
        
        for name, model in text_models:
            print(f"Training {name} (Text) ...")
            probas = cross_val_predict(
                model, x_text, y, cv=self.cv, 
                n_jobs=self.n_jobs, method='predict_proba'
            )

            if meta_x is None:
                meta_x = probas
            else:
                meta_x = np.column_stack((meta_x, probas))

        print("Training Meta Classifier ...")
        self.meta_cls.fit(meta_x, y)

        for name, model in self.speech_models:
            print(f"Re-Training Base {name} (Speech) ...")
            model.fit(x_speech, y)

        for name, model in self.text_models:
            print(f"Re-Training Base {name} (Text) ...")
            model.fit(x_text, y)   

    def predict(self, x_speech, x_text, proba=False):
        meta_x = list()

        for name, model in self.speech_models:
            meta_x.append(model.predict_proba(x_speech))

        for name, model in self.text_models:
            meta_x.append(model.predict_proba(x_text))

        meta_x = np.hstack(meta_x)
        
        return self.meta_cls.predict(meta_x) if not proba else self.meta_cls.predict_proba(meta_x)
