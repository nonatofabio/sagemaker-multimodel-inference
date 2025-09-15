"""
Cleanup TorchServe SageMaker deployment.
"""
import boto3
import os
import shutil

def cleanup_sagemaker(endpoint_name="torchserve-security-models"):
    """Delete SageMaker resources."""
    sm = boto3.client('sagemaker')
    iam = boto3.client('iam')
    
    try:
        # Delete endpoint
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Deleted endpoint: {endpoint_name}")
        
        # Delete endpoint config
        sm.delete_endpoint_config(EndpointConfigName=f"{endpoint_name}-config")
        print(f"✅ Deleted endpoint config: {endpoint_name}-config")
        
        # Delete model
        sm.delete_model(ModelName=endpoint_name)
        print(f"✅ Deleted model: {endpoint_name}")
        
        # Delete the dedicated role
        role_name = "SageMaker-TorchServe-ExecutionRole"
        try:
            iam.detach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
            )
            iam.delete_role(RoleName=role_name)
            print(f"✅ Deleted role: {role_name}")
        except Exception as e:
            print(f"⚠️ Role cleanup: {e}")
        
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

def cleanup_local():
    """Remove local files."""
    dirs_to_remove = ["models", "model-store"]
    files_to_remove = ["model.tar.gz"]
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Removed directory: {dir_name}")
    
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"✅ Removed file: {file_name}")

if __name__ == "__main__":
    print("🧹 Cleaning up TorchServe deployment...")
    cleanup_sagemaker()
    cleanup_local()
    print("✅ Cleanup complete!")
