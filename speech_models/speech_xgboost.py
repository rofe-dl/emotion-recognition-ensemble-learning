from process_dataset.speech_features import get_train_test

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def get_xgb():
    return XGBClassifier(random_state=42)
    # return XGBClassifier(**{'gamma': 0.5, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 1500, 'reg_lambda': 0.8, 'subsample': 0.75, 'random_state': 42})

def main():
    xgb = get_xgb()
    x_train, x_test, y_train, y_test = get_train_test()
    xgb.fit(x_train, y_train)

    results = xgb.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()