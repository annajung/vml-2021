import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath

web_downloader_op = kfp.components.load_component_from_url(
    url='https://raw.githubusercontent.com/kubeflow/pipelines/master/components/contrib/web/Download/component.yaml'
)


def preprocess(file_path: InputPath('Tarball'),
               output_x_train_path: OutputPath('Dataset'),
               output_x_test_path: OutputPath('Dataset'),
               output_y_train_path: OutputPath('Dataset'),
               output_y_test_path: OutputPath('Dataset')
               ):
    import glob
    import pandas as pd
    import tarfile
    from sklearn.model_selection import train_test_split

    tarfile.open(name=file_path, mode="r|gz").extractall('data')
    iris = pd.concat(
        [pd.read_csv(csv_file, header=None)
         for csv_file in glob.glob('data/*.csv')])

    X = iris.iloc[:, :-1]
    y = iris.iloc[:, -1]

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    xtrain.to_csv(output_x_train_path, index=False, header=False)
    xtest.to_csv(output_x_test_path, index=False, header=False)
    ytrain.to_csv(output_y_train_path, index=False, header=False)
    ytest.to_csv(output_y_test_path, index=False, header=False)


def train_model_using_knn(
        x_train: InputPath('Dataset'),
        y_train: InputPath('Dataset'),
        n_neighbors: int,
        model: OutputPath('Model')
):
    import joblib
    import pandas as pd
    from sklearn import neighbors

    x_train_data = pd.read_csv(x_train, header=None)
    y_train_data = pd.read_csv(y_train, header=None)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train_data, y_train_data)

    joblib.dump(classifier, model)


def test_model(
        x_test: InputPath('Dataset'),
        y_test: InputPath('Dataset'),
        model: InputPath('Model')
) -> float:
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


@dsl.pipeline(
    name='iris_pipeline_v1'
)
def iris_pipeline_v1(
        url='https://storage.googleapis.com/ml-pipeline-playground/iris-csv-files.tar.gz',
        n_neighbors: int = 3
):
    web_downloader_task = web_downloader_op(url=url)

    preprocess_op = kfp.components.create_component_from_func(
        func=preprocess,
        base_image='python:3.9-slim',
        packages_to_install=['pandas>=1.3.3', 'scikit-learn']
    )

    preprocess_task = preprocess_op(file=web_downloader_task.outputs['data'])

    train_model_using_knn_op = kfp.components.create_component_from_func(
        func=train_model_using_knn,
        base_image='python:3.9-slim',
        packages_to_install=['pandas>=1.3.3', 'scikit-learn']
    )

    train_model_knn_task = train_model_using_knn_op(x_train=preprocess_task.outputs['output_x_train'],
                                                    y_train=preprocess_task.outputs['output_y_train'],
                                                    n_neighbors=n_neighbors).after(preprocess_task)

    test_model_op = kfp.components.create_component_from_func(
        func=test_model,
        base_image='python:3.9-slim',
        packages_to_install=['pandas>=1.3.3', 'scikit-learn']
    )

    test_model_op(x_test=preprocess_task.outputs['output_x_test'],
                  y_test=preprocess_task.outputs['output_y_test'],
                  model=train_model_knn_task.outputs['model'])


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=iris_pipeline_v1,
        package_path='iris_pipeline_v1.yaml')
