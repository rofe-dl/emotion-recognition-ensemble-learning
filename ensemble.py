from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier

from process_dataset.speech_features import get_speech_features
import numpy as np

import warnings
warnings.filterwarnings('ignore') 

def get_models():

    svm_ = SVC(kernel='linear', probability=True)
    rfc_ = RandomForestClassifier()
    gnb_ = GaussianNB(var_smoothing=0.15199110829529336)
    lr_ = LogisticRegression(solver='liblinear')
    mlp_ = MLPClassifier()

    models = [svm_, rfc_, gnb_, lr_, mlp_]
    model_names = ['Support Vector Machine', 'Random Forest Classifier', 'Gaussian Naive Bayes', 'Logistic Regression', 'MLP Classifier']

    return models, model_names

def stacking_ensemble():

    models, model_names = get_models()

    meta_cls = LogisticRegression()
    stack = StackingClassifier(classifiers=list(models), use_probas=True, meta_classifier=meta_cls)

    models.append(stack)
    model_names.append('STACKED ENSEMBLE')

    data = get_speech_features()
    x = MinMaxScaler().fit_transform(data[0])
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(data[1]), random_state=42, test_size=0.2)

    for model, model_name in zip(models, model_names):
        print(f"{model_name}\n================================")
        model.fit(x_train, y_train)
        results = model.predict(x_test)

        print('Accuracy: ', accuracy_score(y_test, results))
        print(classification_report(y_test, results))

stacking_ensemble()