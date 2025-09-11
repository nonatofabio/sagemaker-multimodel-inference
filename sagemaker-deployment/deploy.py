"""
Main deployment script for HuggingFace models on AWS SageMaker.
Deploys two models as production variants on a single endpoint for GPU isolation.
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import image_uris
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import time
from datetime import datetime
import sys
import json
from config import (
    AWS_REGION, EXECUTION_ROLE_NAME, ENDPOINT_NAME, ENDPOINT_CONFIG_NAME,
    MODELS_CONFIG, INFERENCE_CONFIG,
    ENABLE_DATA_CAPTURE, DATA_CAPTURE_SAMPLING_PERCENTAGE, DEPLOYMENT_TIMEOUT
)


def get_or_create_execution_role():
    """Get the SageMaker execution role or create one if it doesn't exist."""
    iam = boto3.client('iam', region_name=AWS_REGION)
    
    try:
        # Try to get the role
        role = iam.get_role(RoleName=EXECUTION_ROLE_NAME)
        role_arn = role['Role']['Arn']
        print(f"✓ Using existing IAM role: {role_arn}")
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        print(f"→ Creating new IAM role: {EXECUTION_ROLE_NAME}")
        
        # Create the role
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            create_role_response = iam.create_role(
                RoleName=EXECUTION_ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description='SageMaker execution role for HuggingFace model deployment'
            )
            
            # Attach necessary policies
            iam.attach_role_policy(
                RoleName=EXECUTION_ROLE_NAME,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            # Wait for role to be available
            time.sleep(10)
            
            role_arn = create_role_response['Role']['Arn']
            print(f"✓ Created IAM role: {role_arn}")
            return role_arn
            
        except Exception as e:
            print(f"✗ Error creating IAM role: {str(e)}")
            print("  Please create the role manually or use an existing one.")
            sys.exit(1)

def create_huggingface_model(session, role, model_config, model_number):
    """Create a HuggingFace model for deployment."""
    
    print(f"\n{'='*60}")
    print(f"Model {model_number}: {model_config['variant_name']}")
    print(f"{'='*60}")
    
    model_name = f"{model_config['name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Environment variables for the container
    hub_env = {
        'HF_MODEL_ID': model_config['model_id'],
        'HF_TASK': model_config['task'],
        'MAX_INPUT_LENGTH': str(INFERENCE_CONFIG['max_input_length']),
        'MAX_TOTAL_TOKENS': str(INFERENCE_CONFIG['max_total_tokens']),
    }
    
    print(f"→ Creating model: {model_name}")
    print(f"  Model ID: {model_config['model_id']}")
    print(f"  Task: {model_config['task']}")
    print(f"  Instance type: {model_config['instance_type']}")
    
    # Create the HuggingFace model - let HuggingFace SDK automatically select the best image
    # This is the recommended approach from HuggingFace documentation
    huggingface_model = HuggingFaceModel(
        name=model_name,
        env=hub_env,
        role=role,
        transformers_version='4.37',  # Specify transformers version
        pytorch_version='2.1',         # Specify PyTorch version
        py_version='py310',            # Specify Python version
        sagemaker_session=session
    )
    
    # Create the model in SageMaker
    container_def = huggingface_model.prepare_container_def(
        instance_type=model_config['instance_type']
    )
    
    # Register the model
    session.create_model(
        name=model_name,
        role=role,
        container_defs=container_def
    )
    
    print(f"✓ Model created: {model_name}")
    
    return model_name


def create_multi_variant_endpoint_config(session, model_names):
    """Create an endpoint configuration with multiple production variants."""
    
    print(f"\n{'='*60}")
    print("Creating Endpoint Configuration")
    print(f"{'='*60}")
    
    config_name = f"{ENDPOINT_CONFIG_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Create production variants
    production_variants = []
    
    for i, (model_key, model_config) in enumerate(MODELS_CONFIG.items()):
        variant = {
            'VariantName': model_config['variant_name'],
            'ModelName': model_names[i],
            'InitialInstanceCount': model_config['initial_instance_count'],
            'InstanceType': model_config['instance_type'],
            'InitialVariantWeight': model_config['initial_variant_weight']
        }
        production_variants.append(variant)
        
        print(f"→ Adding variant: {model_config['variant_name']}")
        print(f"  Model: {model_names[i]}")
        print(f"  Instances: {model_config['initial_instance_count']} x {model_config['instance_type']}")
        print(f"  Weight: {model_config['initial_variant_weight']}")
    
    # Create endpoint configuration
    sm_client = session.boto_session.client('sagemaker', region_name=AWS_REGION)
    
    # Add data capture configuration if enabled
    config_args = {
        'EndpointConfigName': config_name,
        'ProductionVariants': production_variants
    }
    
    if ENABLE_DATA_CAPTURE:
        config_args['DataCaptureConfig'] = {
            'EnableCapture': True,
            'InitialSamplingPercentage': DATA_CAPTURE_SAMPLING_PERCENTAGE,
            'DestinationS3Uri': f"s3://{session.default_bucket()}/data-capture/{ENDPOINT_NAME}",
            'CaptureOptions': [
                {'CaptureMode': 'Input'},
                {'CaptureMode': 'Output'}
            ],
            'CaptureContentTypeHeader': {
                'JsonContentTypes': ['application/json']
            }
        }
        print(f"\n→ Data capture enabled: {DATA_CAPTURE_SAMPLING_PERCENTAGE}% sampling")
    
    sm_client.create_endpoint_config(**config_args)
    
    print(f"\n✓ Endpoint configuration created: {config_name}")
    
    return config_name


def deploy_endpoint(session, endpoint_config_name):
    """Deploy the endpoint with the given configuration."""
    
    print(f"\n{'='*60}")
    print("Deploying Endpoint")
    print(f"{'='*60}")
    
    endpoint_name = f"{ENDPOINT_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    sm_client = session.boto_session.client('sagemaker', region_name=AWS_REGION)
    
    print(f"→ Creating endpoint: {endpoint_name}")
    print(f"  Configuration: {endpoint_config_name}")
    print(f"  This may take 10-15 minutes...")
    
    # Create the endpoint
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    
    # Wait for endpoint to be in service
    print("\n→ Waiting for endpoint to be ready...")
    
    start_time = time.time()
    timeout = DEPLOYMENT_TIMEOUT
    
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        
        if status == 'InService':
            print(f"\n✓ Endpoint deployed successfully!")
            break
        elif status == 'Failed':
            print(f"\n✗ Endpoint deployment failed!")
            print(f"  Failure reason: {response.get('FailureReason', 'Unknown')}")
            sys.exit(1)
        else:
            elapsed = int(time.time() - start_time)
            print(f"  Status: {status} ({elapsed}s elapsed)", end='\r')
            
            if elapsed > timeout:
                print(f"\n✗ Deployment timeout after {timeout} seconds")
                sys.exit(1)
            
            time.sleep(30)
    
    return endpoint_name


def test_endpoint(endpoint_name, variant_name, test_text, expected_task):
    """Test a specific variant of the endpoint."""
    
    print(f"\n→ Testing variant: {variant_name}")
    print(f"  Task: {expected_task}")
    print(f"  Input: {test_text[:50]}...")
    
    runtime_client = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    try:
        # Prepare the payload
        payload = json.dumps({"inputs": test_text})
        
        # Invoke the endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            TargetVariant=variant_name,
            ContentType='application/json',
            Accept='application/json',
            Body=payload
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        print(f"✓ Variant {variant_name} is working!")
        print(f"  Response preview: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing variant {variant_name}: {str(e)}")
        return False


def main():
    """Main deployment function."""
    
    print("\n" + "="*60)
    print("🚀 SageMaker HuggingFace Multi-Model Deployment")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Region: {AWS_REGION}")
    
    try:
        # Initialize SageMaker session
        print("\n→ Initializing SageMaker session...")
        session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
        print(f"✓ Session initialized")
        print(f"  Default bucket: {session.default_bucket()}")
        
        # Get or create execution role
        role = get_or_create_execution_role()
        
        # Create models
        model_names = []
        for i, (model_key, model_config) in enumerate(MODELS_CONFIG.items(), 1):
            model_name = create_huggingface_model(session, role, model_config, i)
            model_names.append(model_name)
        
        # Create endpoint configuration
        endpoint_config_name = create_multi_variant_endpoint_config(session, model_names)
        
        # Deploy endpoint
        endpoint_name = deploy_endpoint(session, endpoint_config_name)
        
        # Test the endpoint
        print(f"\n{'='*60}")
        print("Testing Endpoint")
        print(f"{'='*60}")
        
        test_cases = [
            {
                'variant': MODELS_CONFIG['pii_masking']['variant_name'],
                'text': "My name is John Doe and my email is john.doe@example.com",
                'task': MODELS_CONFIG['pii_masking']['task']
            },
            {
                'variant': MODELS_CONFIG['prompt_injection']['variant_name'],
                'text': "Please analyze this text for potential security issues.",
                'task': MODELS_CONFIG['prompt_injection']['task']
            }
        ]
        
        all_tests_passed = True
        for test_case in test_cases:
            success = test_endpoint(
                endpoint_name, 
                test_case['variant'], 
                test_case['text'],
                test_case['task']
            )
            all_tests_passed = all_tests_passed and success
        
        # Print summary
        print(f"\n{'='*60}")
        print("✅ Deployment Complete!")
        print(f"{'='*60}")
        print(f"\n📌 Endpoint Details:")
        print(f"  Name: {endpoint_name}")
        print(f"  Region: {AWS_REGION}")
        print(f"  Variants:")
        for model_config in MODELS_CONFIG.values():
            print(f"    - {model_config['variant_name']}: {model_config['model_id']}")
        
        print(f"\n📖 To invoke the endpoint:")
        print(f"  Use boto3 sagemaker-runtime client with:")
        print(f"    EndpointName='{endpoint_name}'")
        print(f"    TargetVariant='PIIMasking' or 'PromptInjection'")
        
        print(f"\n⚠️  To clean up resources, run: python cleanup.py")
        
        # Save deployment info for cleanup
        deployment_info = {
            'endpoint_name': endpoint_name,
            'endpoint_config_name': endpoint_config_name,
            'model_names': model_names,
            'region': AWS_REGION,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"\n💾 Deployment info saved to: deployment_info.json")
        
    except Exception as e:
        print(f"\n✗ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()