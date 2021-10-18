import kfp
from kfp import dsl


def echo_op1(message: str):
    return dsl.ContainerOp(
        name='echo',
        image='library/bash:4.4.23',
        command=['sh', '-c'],
        arguments=['echo ' + str(message)]
    )


@dsl.pipeline(name='hello-world-pipeline-v1')
def hello_world_pipeline_v1(message='hello world'):
    echo_task = echo_op1(message)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline_func=hello_world_pipeline_v1, package_path='hello_world_pipeline_v1.yaml')
