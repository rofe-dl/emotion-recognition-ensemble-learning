from process_dataset.speech_features import get_train_test

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def get_xgb():
    # return XGBClassifier(random_state=42, tree_method='gpu_hist')
    return XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0.5, gpu_id=0, importance_type=None,
              interaction_constraints='', learning_rate=0.05, max_delta_step=0,
              max_depth=3, min_child_weight=1,
              monotone_constraints='()', n_estimators=1500, n_jobs=8,
              num_parallel_tree=1, objective='multi:softprob', predictor='auto',
              random_state=42, reg_alpha=0, reg_lambda=0.8,
              scale_pos_weight=None, subsample=0.75, tree_method='gpu_hist',
              validate_parameters=1, verbosity=None)

def main():
    xgb = get_xgb()
    x_train, x_test, y_train, y_test = get_train_test()
    xgb.fit(x_train, y_train)

    results = xgb.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()