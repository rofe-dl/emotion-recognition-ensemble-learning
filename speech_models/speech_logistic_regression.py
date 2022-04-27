from process_dataset.speech_features import get_train_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def get_logistic_regression():
    # return LogisticRegression(solver='newton-cg', random_state=42)
    return LogisticRegression(C=2, max_iter=2000, penalty='l1', random_state=42, solver='liblinear')

def main():
    lr = get_logistic_regression()
    x_train, x_test, y_train, y_test = get_train_test()
    lr.fit(x_train, y_train)

    results = lr.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()