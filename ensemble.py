
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from process_dataset.speech_features import get_speech_features

from mlxtend.classifier import StackingClassifier
import numpy as np

svm_ = SVC(kernel='linear', probability=True)
rfc_ = RandomForestClassifier()
gnb_ = GaussianNB(var_smoothing=0.15199110829529336)

meta_cls = LogisticRegression()

stack = StackingClassifier(classifiers=[svm_, rfc_, gnb_], use_probas=True, meta_classifier=meta_cls)

models = [svm_, rfc_, gnb_, stack]

data = get_speech_features()
x_train, x_test, y_train, y_test = train_test_split(np.array(data[0]), np.array(data[1]), random_state=42, test_size=0.2)

for model in models:
    model.fit(x_train, y_train)
    results = model.predict(x_test)

    print(accuracy_score(y_test, results))



