"""
Deploy TorchServe model to SageMaker.
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import tarfile
import os
import json
import time
from model_config import SAGEMAKER_CONFIG, TORCHSERVE_CONFIG

def upload_model_to_s3(mar_file_path, bucket_name, key_prefix="torchserve-models"):
    """Upload model archive to S3."""
    s3 = boto3.client('s3')
    
    # Create tar.gz file containing the .mar file
    tar_path = "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(mar_file_path, arcname=os.path.basename(mar_file_path))
    
    # Upload to S3
    key = f"{key_prefix}/{os.path.basename(tar_path)}"
    s3.upload_file(tar_path, bucket_name, key)
    
    model_data_url = f"s3://{bucket_name}/{key}"
    print(f"✅ Model uploaded to: {model_data_url}")
    
    # Cleanup
    os.remove(tar_path)
    
    return model_data_url

def deploy_model(model_data_url, endpoint_name="torchserve-security-models"):
    """Deploy TorchServe model to SageMaker."""
    session = sagemaker.Session()
    
    # Create dedicated SageMaker role
    iam = boto3.client('iam')
    role_name = "SageMaker-TorchServe-ExecutionRole"
    
    try:
        role_response = iam.get_role(RoleName=role_name)
        role = role_response['Role']['Arn']
        print(f"✅ Using existing role: {role}")
    except iam.exceptions.NoSuchEntityException:
        print(f"🔧 Creating SageMaker role: {role_name}")
        
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy)
        )
        
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        
        account_id = boto3.client('sts').get_caller_identity()['Account']
        role = f"arn:aws:iam::{account_id}:role/{role_name}"
        print(f"✅ Created role: {role}")
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_data_url,
        role=role,
        framework_version=SAGEMAKER_CONFIG["framework_version"],
        py_version=SAGEMAKER_CONFIG["py_version"],
        entry_point="handler.py",
        name=f"torchserve-model-{int(time.time())}"
    )
    print(f"✅ PyTorch model created")
    # Deploy to endpoint
    predictor = pytorch_model.deploy(
        initial_instance_count=SAGEMAKER_CONFIG["initial_instance_count"],
        instance_type=SAGEMAKER_CONFIG["instance_type"],
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    
    print(f"✅ Model deployed to endpoint: {endpoint_name}")
    return predictor

def main():
    """Main deployment function."""
    # Check if model archive exists
    mar_file = f"model-store/{TORCHSERVE_CONFIG['model_name']}.mar"
    if not os.path.exists(mar_file):
        print("❌ Model archive not found. Run create_model_archive.py first.")
        return
    
    # Get default bucket
    session = sagemaker.Session()
    bucket = session.default_bucket()
    
    print("📤 Uploading model to S3...")
    model_data_url = upload_model_to_s3(mar_file, bucket)
    
    print("🚀 Deploying to SageMaker...")
    predictor = deploy_model(model_data_url)
    
    print("🧪 Testing deployment...")
    test_data = {
        "text": "John Smith's email is john@example.com",
        "task": "pii_masking"
    }
    
    try:
        result = predictor.predict(test_data)
        print(f"✅ Test successful: {result}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
