from process_dataset.text_features import get_train_test, get_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

def get_logistic_regression():
    # return LogisticRegression(solver='liblinear', random_state=42)
    return LogisticRegression(**{
        'C': 0.615848211066026, 
        'class_weight': None, 
        'dual': False, 
        'fit_intercept': True, 
        'intercept_scaling': 1, 
        'l1_ratio': None, 
        'max_iter': 300, 
        'multi_class': 'auto', 
        'n_jobs': None, 
        'penalty': 'l2',
        'solver': 'liblinear', 
        'tol': 0.2, 
        'verbose': 0, 
        'warm_start': False,
        'random_state': 42})

def main():
    lr = get_logistic_regression()
    x_train, x_test, y_train, y_test = get_train_test()
    lr.fit(x_train, y_train)

    results = lr.predict(x_test)
    print(classification_report(y_test, results, digits=4))

def k_fold():
    pipeline = Pipeline(
        [('transformer', CountVectorizer(min_df=5, ngram_range=(1, 2))), ('estimator', get_logistic_regression())]
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