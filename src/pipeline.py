import os
import json
import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["S3_BUCKET"]
session = PipelineSession(
    boto_session=boto3.Session(region_name=region),
    default_bucket=bucket,
)

processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=session,
)
process_args = processor.run(
    code="src/preprocess.py",
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
)
preprocess = ProcessingStep(name="PreprocessData", step_args=process_args)

def output_uri(name):
    return preprocess.properties.ProcessingOutputConfig.Outputs[name].S3Output.S3Uri

train_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=session,
)
train_args = train_processor.run(
    code="src/train.py",
    inputs=[
        ProcessingInput(source=output_uri("train"), destination="/opt/ml/processing/train"),
        ProcessingInput(source=output_uri("test"), destination="/opt/ml/processing/test"),
    ],
    outputs=[
        ProcessingOutput(output_name="model", source="/opt/ml/processing/model"),
    ],
)
training = ProcessingStep(name="TrainModel", step_args=train_args)

model_artifact_uri = Join(
    on="/",
    values=[
        training.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
        "model.tar.gz",
    ],
)

model = SKLearnModel(
    model_data=model_artifact_uri,
    role=role,
    entry_point="inference.py",
    source_dir="src",
    code_location=f"s3://{bucket}/code",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)
register_args = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.c7i.large", "ml.m5d.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="CancerClassificationModels",
    approval_status="Approved",
)

registration = ModelStep(name="RegisterModel", step_args=register_args)

pipeline = Pipeline(
    name="CancerClassificationPipeline",
    steps=[preprocess, training, registration],
    sagemaker_session=session,
)
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Execution ARN:", execution.arn, flush=True)
with open("execution.json", "w") as file:
    json.dump({"execution_arn": execution.arn}, file)
execution.wait()
status = execution.describe()["PipelineExecutionStatus"]
if status != "Succeeded":
    raise RuntimeError(f"Pipeline failed: {status}")
print("Pipeline succeeded")
