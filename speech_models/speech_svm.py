from process_dataset.speech_features import get_train_test

from sklearn.svm import SVC
from sklearn.metrics import classification_report

def get_svm():
    return SVC(kernel='linear', probability=True, random_state=42)

def main():
    svm = get_svm()
    x_train, x_test, y_train, y_test = get_train_test()
    svm.fit(x_train, y_train)

    results = svm.predict(x_test)
    print(classification_report(y_test, results))

if __name__ == '__main__':
    main()