import kfp
import kfp.v2.dsl as dsl
from kfp.v2.dsl import (
    component
)


@component(base_image='python:3.9-slim')
def echo_op(message: str):
    print(message)


@dsl.pipeline(name='hello-world-pipeline-v2')
def hello_world_pipeline_v2(message: str = 'hello world'):
    echo_task = echo_op(message)


if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE) \
        .compile(pipeline_func=hello_world_pipeline_v2, package_path='hello_world_pipeline_v2.yaml')
