from process_dataset.text_features import get_train_test

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def get_xgb():
    return XGBClassifier(random_state=42, tree_method='gpu_hist')

def main():
    xgb = get_xgb()
    x_train, x_test, y_train, y_test = get_train_test()
    xgb.fit(x_train, y_train)

    results = xgb.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()