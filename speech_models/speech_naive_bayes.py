from process_dataset.speech_features import get_train_test

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report

def get_naive_bayes():
    # return MultinomialNB(alpha=0.01)
    return MultinomialNB(alpha=4.2)

def main():
    mnb = get_naive_bayes()
    x_train, x_test, y_train, y_test = get_train_test()
    mnb.fit(x_train, y_train)

    results = mnb.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()