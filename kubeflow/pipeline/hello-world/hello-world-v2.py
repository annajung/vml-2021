import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (
    component
)


@component(base_image='library/bash:4.4.23')
def echo_op():
    print("Hello world")


@dsl.pipeline(name='hello-world-pipeline-v2')
def hello_world_pipeline_v2():
    echo_task = echo_op()


if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE) \
        .compile(pipeline_func=hello_world_pipeline_v2, package_path='hello_world_pipeline_v2.yaml')
