from process_dataset.speech_features import get_train_test, get_data

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np

def get_xgb():
    # return XGBClassifier(random_state=42)
    return XGBClassifier(**{'gamma': 0.5, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 1500, 'reg_lambda': 0.8, 'subsample': 0.75, 'random_state': 42, 'tree_method': 'gpu_hist'})

def main():
    xgb = get_xgb()
    x_train, x_test, y_train, y_test = get_train_test()
    xgb.fit(x_train, y_train)

    results = xgb.predict(x_test)
    print(classification_report(y_test, results, digits=4))

def k_fold():
    pipeline = Pipeline(
        [('transformer', MinMaxScaler()), ('estimator', get_xgb())]
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