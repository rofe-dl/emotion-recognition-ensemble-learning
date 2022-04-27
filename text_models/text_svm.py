from process_dataset.text_features import get_train_test

from sklearn.svm import SVC
from sklearn.metrics import classification_report

def get_svm():
    return SVC(kernel='linear', probability=True, random_state=42)
    # return SVC(kernel='rbf', gamma='scale', C=1.85, probability=True, random_state=42)

def main():
    svm = get_svm()
    x_train, x_test, y_train, y_test = get_train_test()
    svm.fit(x_train, y_train)

    results = svm.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()