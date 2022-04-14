import numpy as np
import copy
from multiprocessing import Pool

class StackEnsembleCustom():
    
    def __init__ (self, speech_models, text_models, meta_cls, cv):
        self.speech_models = speech_models
        self.text_models = text_models
        self.meta_cls = meta_cls
        self.cv = cv

    def fit(self, x_speech, x_text, y):
        # TODO write logs
        meta_x = list()

        x_speech_split = np.array_split(x_speech, self.cv)
        x_text_split = np.array_split(x_text, self.cv)
        y_split = np.array_split(y, self.cv)

        for i in range(self.cv):

            speech_models = copy.deepcopy(self.speech_models)
            text_models = copy.deepcopy(self.text_models)

            x_speech_train, x_speech_test, x_text_train, x_text_test, y_train, y_test = self._get_train_test(x_speech_split, x_text_split, y_split, i)
            probas = list()

            for name, model in speech_models:
                model.fit(x_speech_train, y_train)
                probas.append(model.predict_proba(x_speech_test))

            for name, model in text_models:
                model.fit(x_text_train, y_train)
                probas.append(model.predict_proba(x_text_test))

            probas = np.hstack(probas)
            meta_x.extend(probas)
        
        self.meta_cls.fit(meta_x, y)

        # retrain all base models, not the deepcopies

        for name, model in self.speech_models:
            model.fit(x_speech, y)

        for name, model in self.text_models:
            model.fit(x_text, y)

    def predict(self, x_speech, x_text):
        meta_x = list()

        for name, model in self.speech_models:
            meta_x.append(model.predict_proba(x_speech))

        for name, model in self.text_models:
            meta_x.append(model.predict_proba(x_text))

        meta_x = np.hstack(meta_x)
        
        return self.meta_cls.predict(meta_x)


    def _get_train_test(self, x_speech_split, x_text_split, y_split, i):
        x_speech_test = x_speech_split[i]
        x_text_test = x_text_split[i]
        y_test = y_split[i]

        x_speech_train = np.concatenate([split for index, split in enumerate(x_speech_split) if index != i])
        x_text_train = np.concatenate([split for index, split in enumerate(x_text_split) if index != i])
        y_train = np.concatenate([split for index, split in enumerate(y_split) if index != i])

        return x_speech_train, x_speech_test, x_text_train, x_text_test, y_train, y_test

        