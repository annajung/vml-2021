import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Model,
    Artifact,
    OutputPath,
    InputPath,
    Output,
    ClassificationMetrics,
    Condition
)

# Web downloader component - Can be reused anywhere
web_downloader_op = kfp.components.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/1.7.0/components/web/Download/component-sdk-v2.yaml')


@component(
    packages_to_install=['pandas>=1.3.3', 'scikit-learn'],
    base_image='python:3.9-slim'
)
def preprocess_op(
        tar_data: Input[Artifact],
        output_x_train_path: OutputPath(Artifact),
        output_x_test_path: OutputPath(Artifact),
        output_y_train_path: OutputPath(Artifact),
        output_y_test_path: OutputPath(Artifact)
):
    import glob
    import tarfile
    import pandas as pd
    from sklearn.model_selection import train_test_split

    tarfile.open(name=tar_data.path, mode="r|gz").extractall('data')
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


@component(
    packages_to_install=['pandas>=1.3.3', 'scikit-learn'],
    base_image='python:3.9-slim'
)
def train_model_using_decision_tree_op(
        x_train: InputPath(Artifact),
        y_train: InputPath(Artifact),
        splitter: str,
        model: OutputPath(Model)
):
    import joblib
    import pandas as pd
    from sklearn import tree

    x_train_data = pd.read_csv(x_train, header=None)
    y_train_data = pd.read_csv(y_train, header=None)

    classifier = tree.DecisionTreeClassifier(splitter=splitter)
    classifier.fit(x_train_data, y_train_data)

    joblib.dump(classifier, model)


@component(
    packages_to_install=['pandas>=1.3.3', 'scikit-learn'],
    base_image='python:3.9-slim'
)
def train_model_using_knn_op(
        x_train: InputPath(Artifact),
        y_train: InputPath(Artifact),
        n_neighbors: int,
        model: OutputPath(Model)
):
    import joblib
    import pandas as pd
    from sklearn import neighbors

    x_train_data = pd.read_csv(x_train, header=None)
    y_train_data = pd.read_csv(y_train, header=None)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train_data, y_train_data)

    joblib.dump(classifier, model)


@component(
    packages_to_install=['pandas>=1.3.3', 'scikit-learn'],
    base_image='python:3.9-slim'
)
def test_model_op(
        x_test: InputPath(Artifact),
        y_test: InputPath(Artifact),
        model: InputPath(Model)
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


@component(
    packages_to_install=['pandas>=1.3.3', 'scikit-learn'],
    base_image='python:3.9-slim'
)
def confusion_matrix_op(
        x_test: InputPath(Artifact),
        y_test: InputPath(Artifact),
        model: InputPath(Model),
        metrics: Output[ClassificationMetrics]
):
    import joblib
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    x_test_data = pd.read_csv(x_test, header=None)
    y_test_data = pd.read_csv(y_test, header=None)

    loaded_model = joblib.load(model)
    predictions = loaded_model.predict(x_test_data)

    metrics.log_confusion_matrix(
        ['Setosa', 'Versicolour', 'Virginica'],
        confusion_matrix(y_test_data, predictions).tolist()
    )


@component(
    packages_to_install=['pandas>=1.3.3', 'scikit-learn', 'minio'],
    base_image='python:3.9-slim'
)
def save_model_op(
        model: InputPath(Model),
        model_name: str
) -> str:
    import joblib
    from minio import Minio

    print(model_name)

    loaded_model = joblib.load(model)
    joblib.dump(loaded_model, "temp.joblib")

    # please dO NOT hardcode your access key and secret key!
    client = Minio(
        "minio-service.kubeflow:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )

    bucket_name = "v2-model"
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
    else:
        print("Bucket {} already exists".format(bucket_name))

    client.fput_object(bucket_name, "model.joblib", "./temp.joblib")

    return "s3://{}/{}".format(bucket_name, "model.joblib")


@dsl.pipeline(
    name='iris_pipeline_v2'
)
def iris_pipeline_v2(
        url: str = 'https://storage.googleapis.com/ml-pipeline-playground/iris-csv-files.tar.gz',
        n_neighbors: int = 3,
        splitter: str = 'random'
):
    web_downloader_task = web_downloader_op(url=url)

    preprocess_task = preprocess_op(tar_data=web_downloader_task.outputs['data'])

    train_model_knn_task = train_model_using_knn_op(
        x_train=preprocess_task.outputs['output_x_train_path'],
        y_train=preprocess_task.outputs['output_y_train_path'],
        n_neighbors=n_neighbors
    )

    train_model_tree_task = train_model_using_decision_tree_op(
        x_train=preprocess_task.outputs['output_x_train_path'],
        y_train=preprocess_task.outputs['output_y_train_path'],
        splitter=splitter
    )

    test_model_knn_task = test_model_op(
        x_test=preprocess_task.outputs['output_x_test_path'],
        y_test=preprocess_task.outputs['output_y_test_path'],
        model=train_model_knn_task.outputs['model']
    )

    test_model_tree_task = test_model_op(
        x_test=preprocess_task.outputs['output_x_test_path'],
        y_test=preprocess_task.outputs['output_y_test_path'],
        model=train_model_tree_task.outputs['model']
    )

    confusion_matrix_knn_task = confusion_matrix_op(
        x_test=preprocess_task.outputs['output_x_test_path'],
        y_test=preprocess_task.outputs['output_y_test_path'],
        model=train_model_knn_task.outputs['model']
    )

    confusion_matrix_tree_task = confusion_matrix_op(
        x_test=preprocess_task.outputs['output_x_test_path'],
        y_test=preprocess_task.outputs['output_y_test_path'],
        model=train_model_tree_task.outputs['model']
    )

    with Condition(test_model_knn_task.output >= test_model_tree_task.output, 'knn'):
        save_model_knn_task = save_model_op(model=train_model_knn_task.outputs['model'], model_name="knn")

    with Condition(test_model_tree_task.output > test_model_knn_task.output, 'tree'):
        save_model_tree_task = save_model_op(model=train_model_tree_task.outputs['model'], model_name="tree")


if __name__ == "__main__":
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=iris_pipeline_v2,
        package_path='iris_pipeline_v2.yaml')
