import numpy as np
import copy
import os
import pickle

from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

class Ensemble():

    def save(self, file_name):
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')

        with open(f"trained_models/{file_name}", 'wb') as f:
            pickle.dump(self, f)

    def cross_validate(self, x_speech, x_text, y, cv):

        if len(x_speech) < cv:
            raise ValueError("Dataset is too small for k")

        accuracies, f1s_macro, f1s_weighted, precisions_macro, precisions_weighted, recalls_macro, recalls_weighted = (list() for i in range(7))
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
            f1s_macro.append(f1_score(y_test, result, average='macro'))
            precisions_macro.append(precision_score(y_test, result, average='macro'))
            recalls_macro.append(recall_score(y_test, result, average='macro'))
            f1s_weighted.append(f1_score(y_test, result, average='weighted'))
            precisions_weighted.append(precision_score(y_test, result, average='weighted'))
            recalls_weighted.append(recall_score(y_test, result, average='weighted'))


        return {
            'test_accuracy': accuracies,
            'test_f1_macro': f1s_macro,
            'test_precision_macro': precisions_macro,
            'test_recall_macro': recalls_macro,
            'test_f1_weighted': f1s_weighted,
            'test_precision_weighted': precisions_weighted,
            'test_recall_weighted': recalls_weighted
        }

class VoteEnsemble(Ensemble):

    def __init__(self, speech_models, text_models, type):
        self.speech_models = speech_models
        self.text_models = text_models
        self.type = type

        if type == 'soft':
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
        if self.type == 'soft':
            self.voter_speech.fit(x_speech, y)
            self.voter_text.fit(x_text, y)
        else:
            for name, model in self.speech_models:
                print(f"Training {name} (Speech) ...")
                model.fit(x_speech, y)

            for name, model in self.text_models:
                print(f"Training {name} (Text) ...")
                model.fit(x_text, y)

    def predict(self, x_speech, x_text, proba=False):
        if self.type == 'soft':
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

        else:

            if proba:
                raise ValueError("When using hard voting, proba can't be True")

            all_predictions = []
            for name, model in self.speech_models:
                all_predictions.append(list(model.predict(x_speech)))

            for name, model in self.text_models:
                all_predictions.append(list(model.predict(x_text)))

            all_predictions = list(zip(*all_predictions))
            result = self._find_majority(all_predictions)

        return result

    def _find_majority(self, all_predictions):

        result = []
        for prediction in all_predictions:
            max_count = 0
            max_emotion = prediction[0]

            for emotion in prediction:
                count = prediction.count(emotion)
                
                if count > max_count:
                    max_count = count
                    max_emotion = emotion
            
            result.append(max_emotion)

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
