from process_dataset.speech_features import get_train_test

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def get_mlp():
    return MLPClassifier(random_state=42, max_iter=5000)
    # return MLPClassifier(alpha=0.07, hidden_layer_sizes=(140, 100, 60),
    #           learning_rate='invscaling', max_iter=100, random_state=42)

def main():
    mlp = get_mlp()
    x_train, x_test, y_train, y_test = get_train_test()
    mlp.fit(x_train, y_train)

    results = mlp.predict(x_test)
    print(classification_report(y_test, results, digits=4))

if __name__ == '__main__':
    main()