from process_dataset.speech_features import get_train_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def get_logistic_regression():
    # return LogisticRegression(random_state=42)
    return LogisticRegression(solver='liblinear', random_state=42, penalty='l2', C=1.58)

def main():
    lr = get_logistic_regression()
    x_train, x_test, y_train, y_test = get_train_test()
    lr.fit(x_train, y_train)

    results = lr.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()