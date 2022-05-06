from process_dataset.speech_features import get_train_test, get_data

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np

def get_svm():
    # return SVC(kernel='linear', probability=True, random_state=42)
    return SVC(
        C=20, decision_function_shape='ovo', degree=7, probability=True,
        random_state=42, shrinking=False)

def main():
    svm = get_svm()
    x_train, x_test, y_train, y_test = get_train_test()
    svm.fit(x_train, y_train)

    results = svm.predict(x_test)
    print(classification_report(y_test, results, digits=4))

def k_fold():
    pipeline = Pipeline(
        [('transformer', MinMaxScaler()), ('estimator', get_svm())]
    )
    
    x, y = get_data()
    scoring = {'accuracy': 'accuracy',
           'f1_macro': 'f1_macro',
           'precision_macro': 'precision_macro',
           'recall_macro' : 'recall_macro',
           'f1_weighted' : 'f1_weighted',
           'precision_weighted' : 'precision_weighted',
           'recall_weighted' : 'recall_weighted'
           }
    
    scores = cross_validate(pipeline, x, y, cv=5, scoring=scoring, n_jobs=-1)

    print('Accuracy: ', np.mean(scores['test_accuracy']))
    print('F1 Macro: ', np.mean(scores['test_f1_macro']))
    print('Precision Macro: ', np.mean(scores['test_precision_macro']))
    print('Recall Macro: ', np.mean(scores['test_recall_macro']))
    print('F1 Weighted: ', np.mean(scores['test_f1_weighted']))
    print('Precision Weighted: ', np.mean(scores['test_precision_weighted']))
    print('Recall Weighted: ', np.mean(scores['test_recall_weighted']))

if __name__ == '__main__':
    print('On Train/Test Split', end='\n\n')
    main()
    print('On K-fold Cross Validation', end='\n\n')
    k_fold()