from process_dataset.text_features import get_train_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def get_random_forest():
    # return RandomForestClassifier(random_state=42)
    return RandomForestClassifier(**{
        'random_state': 42,
        'bootstrap': True, 
        'ccp_alpha': 0.0, 
        'class_weight': None, 
        'criterion': 'gini', 
        'max_depth': 2000, 
        'max_features': 'auto', 
        'max_leaf_nodes': None, 
        'max_samples': None, 
        'min_impurity_decrease': 0.0, 
        'min_samples_leaf': 2, 
        'min_samples_split': 2, 
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 733, 
        'n_jobs': None, 
        'oob_score': False,
        'verbose': 0, 
        'warm_start': False
    })

def main():
    rfc = get_random_forest()
    x_train, x_test, y_train, y_test = get_train_test()
    rfc.fit(x_train, y_train)

    results = rfc.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()