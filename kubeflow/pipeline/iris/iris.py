import glob
import tarfile
import urllib.request

import joblib
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def download_and_merge_csv(url, output_csv):
    with urllib.request.urlopen(url) as res:
        tarfile.open(fileobj=res, mode="r|gz").extractall('data')
    df = pd.concat(
        [pd.read_csv(csv_file, header=None)
         for csv_file in glob.glob('data/*.csv')])
    df.to_csv(output_csv, index=False, header=False)


def preprocess_op(
        tar_data,
        output_x_train_path,
        output_x_test_path,
        output_y_train_path,
        output_y_test_path
):
    iris = pd.read_csv(tar_data, header=None)

    X = iris.iloc[:, :-1]
    y = iris.iloc[:, -1]

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    xtrain.to_csv(output_x_train_path, index=False, header=False)
    xtest.to_csv(output_x_test_path, index=False, header=False)
    ytrain.to_csv(output_y_train_path, index=False, header=False)
    ytest.to_csv(output_y_test_path, index=False, header=False)


def train_model_using_knn_op(
        x_train,
        y_train,
        n_neighbors,
        model
):
    x_train_data = pd.read_csv(x_train, header=None)
    y_train_data = pd.read_csv(y_train, header=None)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train_data, y_train_data)

    joblib.dump(classifier, model)


def test_model_op(
        x_test,
        y_test,
        model
) -> float:
    x_test_data = pd.read_csv(x_test, header=None)
    y_test_data = pd.read_csv(y_test, header=None)

    loaded_model = joblib.load(model)
    predictions = loaded_model.predict(x_test_data)
    accuracy = accuracy_score(y_test_data, predictions)
    print("accuracy={}".format(accuracy))

    return accuracy.item()


if __name__ == "__main__":
    x_train = "data/x_train.csv"
    x_test = "data/x_test.csv"
    y_train = "data/y_train.csv"
    y_test = "data/y_test.csv"
    model = "data/model.joblib"
    url = "https://storage.googleapis.com/ml-pipeline-playground/iris-csv-files.tar.gz"
    output_data = "data/iris.csv"

    download_and_merge_csv(
        url=url,
        output_csv=output_data
    )

    preprocess_op(
        tar_data=output_data,
        output_x_train_path=x_train,
        output_x_test_path=x_test,
        output_y_train_path=y_train,
        output_y_test_path=y_test
    )

    train_model_using_knn_op(
        x_train=x_train,
        y_train=y_train,
        n_neighbors=3,
        model=model
    )

    test_model_op(
        x_test=x_test,
        y_test=y_test,
        model=model
    )
