from process_dataset.text_features import get_train_test

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def get_mlp():
    # return MLPClassifier(random_state=42, max_iter=5000)
    return MLPClassifier(
        random_state=42, activation='relu',
        alpha=0.35,hidden_layer_sizes=(150,),
        learning_rate='adaptive', solver='adam')

def main():
    mlp = get_mlp()
    x_train, x_test, y_train, y_test = get_train_test()
    mlp.fit(x_train, y_train)

    results = mlp.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()