from sklearn import datasets


def preprocess(
        output_x_train_path="/tmp/x-train.csv",
        output_x_test_path="/tmp/x-test.csv",
        output_y_train_path="/tmp/y-train.csv",
        output_y_test_path="/tmp/y-test.csv"
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    iris_dataset = datasets.load_iris()

    X = iris_dataset.data
    y = iris_dataset.target

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    pd.DataFrame(xtrain).to_csv(output_x_train_path, index=False, header=False)
    pd.DataFrame(xtest).to_csv(output_x_test_path, index=False, header=False)
    pd.DataFrame(ytrain).to_csv(output_y_train_path, index=False, header=False)
    pd.DataFrame(ytest).to_csv(output_y_test_path, index=False, header=False)


def train_model_using_decision_tree(x_train, y_train, parameter, model="/tmp/tree.joblib"):
    import joblib
    import pandas as pd
    from sklearn import tree

    x_train_data = pd.read_csv(x_train, header=None)
    y_train_data = pd.read_csv(y_train, header=None)

    classifier = tree.DecisionTreeClassifier(splitter=parameter)
    classifier.fit(x_train_data, y_train_data)

    joblib.dump(classifier, model)


def train_model_using_knn(x_train, y_train, parameter, model="/tmp/knn.joblib"):
    import joblib
    import pandas as pd
    from sklearn import neighbors

    x_train_data = pd.read_csv(x_train, header=None)
    y_train_data = pd.read_csv(y_train, header=None)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=int(parameter))
    classifier.fit(x_train_data, y_train_data)

    joblib.dump(classifier, model)


def test_model(x_test, y_test, model="/tmp/tree.joblib") -> float:
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    x_test_data = pd.read_csv(x_test, header=None)
    y_test_data = pd.read_csv(y_test, header=None)

    loaded_model = joblib.load(model)
    predictions = loaded_model.predict(x_test_data)
    accuracy = accuracy_score(y_test_data, predictions)
    print("accuracy={}".format(accuracy))

    return accuracy


def save_model(model, model_name):
    print(model_name)
    return model


def train_tree(parameter):
    preprocess()
    train_model_using_decision_tree("/tmp/x-train.csv", "/tmp/y-train.csv", parameter, model="/tmp/tree.joblib")


def train_knn(parameter):
    preprocess()
    train_model_using_knn("/tmp/x-train.csv", "/tmp/y-train.csv", parameter, model="/tmp/knn.joblib")


def test(model_name):
    test_model("/tmp/x-test.csv", "/tmp/y-test.csv", model="/tmp/{}.joblib".format(model_name))


if __name__ == '__main__':
    import sys
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_algorithm", type=str, default="knn",
                    help="The training function to be used", choices=["tree", "knn"], required=True)

    ap.add_argument("--n_neighbors", type=int,  default=3,
                    help="Number of neighbors used in KNN algorithm", required=False)

    ap.add_argument("--splitter", type=str, default="random",
                    help="The strategy used to choose the split at each node of the Decision Tree. "
                         "Supported strategies are 'best' and 'random'.", choices=["random", "best"], required=False)
    args = vars(ap.parse_args())

    if args['train_algorithm'] == "tree":
        train_tree(args['splitter'])
        test("tree")
    else:
        train_knn(args['n_neighbors'])
        test("knn")