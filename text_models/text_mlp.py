from process_dataset.text_features import get_train_test, get_data

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

def get_mlp():
    # return MLPClassifier(random_state=42, max_iter=5000)
    return MLPClassifier(
        random_state=42, activation='relu',
        alpha=0.35,hidden_layer_sizes=(150,),
        learning_rate='adaptive', solver='adam')

def main():
    mlp = get_mlp()
    x_train, x_test, y_train, y_test = get_train_test()
    mlp.fit(x_train, y_train)

    results = mlp.predict(x_test)
    print(classification_report(y_test, results, digits=4))

def k_fold():
    pipeline = Pipeline(
        [('transformer', CountVectorizer(min_df=5, ngram_range=(1, 2))), ('estimator', get_mlp())]
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