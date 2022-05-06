from process_dataset.speech_features import get_train_test, get_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np

def get_random_forest():
    # return RandomForestClassifier(random_state=42)
    return RandomForestClassifier(bootstrap=False, max_depth=120, max_features=0.3,
                       min_samples_split=10, n_estimators=1500, n_jobs=-1,
                       random_state=42)

def main():
    rfc = get_random_forest()
    x_train, x_test, y_train, y_test = get_train_test()
    rfc.fit(x_train, y_train)

    results = rfc.predict(x_test)
    print(classification_report(y_test, results, digits=4))

def k_fold():
    pipeline = Pipeline(
        [('transformer', MinMaxScaler()), ('estimator', get_random_forest())]
    )
    
    x, y = get_data()
    scoring = {'accuracy': 'accuracy',
           'f1_macro': 'f1_macro',
           'precision_macro': 'precision_macro',
           'recall_macro' : 'recall_macro'}
    
    scores = cross_validate(pipeline, x, y, cv=5, scoring=scoring, n_jobs=-1)

    print('Accuracy: ', np.mean(scores['test_accuracy']))
    print('F1 Macro: ', np.mean(scores['test_f1_macro']))
    print('Precision Macro: ', np.mean(scores['test_precision_macro']))
    print('Recall Macro: ', np.mean(scores['test_recall_macro']))

if __name__ == '__main__':
    print('On Train/Test Split', end='\n\n')
    main()
    print('On K-fold Cross Validation', end='\n\n')
    k_fold()