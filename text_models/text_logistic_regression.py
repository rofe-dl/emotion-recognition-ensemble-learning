from process_dataset.text_features import get_train_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

if __name__ == '__main__':
    main()