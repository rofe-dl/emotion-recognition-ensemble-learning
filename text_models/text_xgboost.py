from process_dataset.text_features import get_train_test

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def get_xgb():
    # return XGBClassifier(random_state=42, tree_method='gpu_hist')
    return XGBClassifier(**{
        'colsample_bytree': 0.8, 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'reg_lambda': 0.001, 'subsample': 0.4, 'random_state': 42
    })

def main():
    xgb = get_xgb()
    x_train, x_test, y_train, y_test = get_train_test()
    xgb.fit(x_train, y_train)

    results = xgb.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()