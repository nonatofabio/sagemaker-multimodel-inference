"""
Cleanup script for removing SageMaker resources created during deployment.
Deletes endpoint, endpoint configuration, and models.
"""

import boto3
import json
import sys
import time
from datetime import datetime
from config import AWS_REGION, ENDPOINT_NAME


def load_deployment_info():
    """Load deployment information from the saved JSON file."""
    try:
        with open('deployment_info.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("✗ deployment_info.json not found.")
        print("  Looking for resources by name pattern...")
        return None
    except json.JSONDecodeError:
        print("✗ Error parsing deployment_info.json")
        return None


def find_endpoints_by_pattern(sm_client, pattern):
    """Find endpoints that match the given pattern."""
    endpoints = []
    
    try:
        paginator = sm_client.get_paginator('list_endpoints')
        
        for page in paginator.paginate(NameContains=pattern):
            for endpoint in page['Endpoints']:
                endpoints.append(endpoint['EndpointName'])
        
        return endpoints
    except Exception as e:
        print(f"✗ Error listing endpoints: {str(e)}")
        return []


def delete_endpoint(sm_client, endpoint_name):
    """Delete a SageMaker endpoint."""
    print(f"\n→ Deleting endpoint: {endpoint_name}")
    
    try:
        # Check if endpoint exists
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        
        # Delete the endpoint
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        
        # Wait for deletion
        print("  Waiting for endpoint deletion...")
        start_time = time.time()
        
        while True:
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
                elapsed = int(time.time() - start_time)
                print(f"  Still deleting... ({elapsed}s elapsed)", end='\r')
                time.sleep(10)
            except sm_client.exceptions.EndpointNotFound:
                print(f"\n✓ Endpoint deleted: {endpoint_name}")
                break
            except Exception as e:
                print(f"\n✗ Error checking endpoint status: {str(e)}")
                break
                
            if time.time() - start_time > 600:  # 10 minute timeout
                print(f"\n⚠ Timeout waiting for endpoint deletion")
                break
                
        return True
        
    except sm_client.exceptions.EndpointNotFound:
        print(f"  Endpoint not found: {endpoint_name}")
        return False
    except Exception as e:
        print(f"✗ Error deleting endpoint: {str(e)}")
        return False


def delete_endpoint_config(sm_client, config_name):
    """Delete a SageMaker endpoint configuration."""
    print(f"\n→ Deleting endpoint configuration: {config_name}")
    
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"✓ Endpoint configuration deleted: {config_name}")
        return True
    except sm_client.exceptions.EndpointConfigNotFound:
        print(f"  Endpoint configuration not found: {config_name}")
        return False
    except Exception as e:
        print(f"✗ Error deleting endpoint configuration: {str(e)}")
        return False


def delete_model(sm_client, model_name):
    """Delete a SageMaker model."""
    print(f"\n→ Deleting model: {model_name}")
    
    try:
        sm_client.delete_model(ModelName=model_name)
        print(f"✓ Model deleted: {model_name}")
        return True
    except sm_client.exceptions.ModelNotFound:
        print(f"  Model not found: {model_name}")
        return False
    except Exception as e:
        print(f"✗ Error deleting model: {str(e)}")
        return False


def find_and_delete_related_resources(sm_client, base_name):
    """Find and delete all resources related to the base endpoint name."""
    print(f"\n→ Searching for resources with pattern: {base_name}")
    
    deleted_resources = {
        'endpoints': [],
        'configs': [],
        'models': []
    }
    
    # Find and delete endpoints
    endpoints = find_endpoints_by_pattern(sm_client, base_name)
    print(f"  Found {len(endpoints)} endpoint(s)")
    
    for endpoint in endpoints:
        if delete_endpoint(sm_client, endpoint):
            deleted_resources['endpoints'].append(endpoint)
            
            # Get the endpoint configuration name
            try:
                response = sm_client.describe_endpoint(EndpointName=endpoint)
                config_name = response.get('EndpointConfigName')
                if config_name:
                    # Delete the configuration
                    if delete_endpoint_config(sm_client, config_name):
                        deleted_resources['configs'].append(config_name)
                    
                    # Get model names from the configuration
                    try:
                        config_response = sm_client.describe_endpoint_config(
                            EndpointConfigName=config_name
                        )
                        for variant in config_response.get('ProductionVariants', []):
                            model_name = variant.get('ModelName')
                            if model_name:
                                if delete_model(sm_client, model_name):
                                    deleted_resources['models'].append(model_name)
                    except:
                        pass
            except:
                pass
    
    # Also search for models and configs by pattern
    try:
        # Search for models
        paginator = sm_client.get_paginator('list_models')
        for page in paginator.paginate(NameContains=base_name):
            for model in page['Models']:
                model_name = model['ModelName']
                if model_name not in deleted_resources['models']:
                    if delete_model(sm_client, model_name):
                        deleted_resources['models'].append(model_name)
        
        # Search for configurations
        paginator = sm_client.get_paginator('list_endpoint_configs')
        for page in paginator.paginate(NameContains=base_name):
            for config in page['EndpointConfigs']:
                config_name = config['EndpointConfigName']
                if config_name not in deleted_resources['configs']:
                    if delete_endpoint_config(sm_client, config_name):
                        deleted_resources['configs'].append(config_name)
    except Exception as e:
        print(f"⚠ Error searching for additional resources: {str(e)}")
    
    return deleted_resources


def main():
    """Main cleanup function."""
    
    print("\n" + "="*60)
    print("🧹 SageMaker Deployment Cleanup")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Region: {AWS_REGION}")
    
    # Initialize SageMaker client
    sm_client = boto3.client('sagemaker', region_name=AWS_REGION)
    
    # Load deployment info
    deployment_info = load_deployment_info()
    
    deleted_resources = {
        'endpoints': [],
        'configs': [],
        'models': []
    }
    
    if deployment_info:
        print(f"\n📋 Found deployment info from: {deployment_info.get('timestamp', 'unknown')}")
        
        # Delete endpoint
        endpoint_name = deployment_info.get('endpoint_name')
        if endpoint_name:
            if delete_endpoint(sm_client, endpoint_name):
                deleted_resources['endpoints'].append(endpoint_name)
        
        # Delete endpoint configuration
        config_name = deployment_info.get('endpoint_config_name')
        if config_name:
            if delete_endpoint_config(sm_client, config_name):
                deleted_resources['configs'].append(config_name)
        
        # Delete models
        model_names = deployment_info.get('model_names', [])
        for model_name in model_names:
            if delete_model(sm_client, model_name):
                deleted_resources['models'].append(model_name)
    else:
        # Try to find resources by pattern
        print(f"\n→ No deployment info file found.")
        
        # Ask user for confirmation
        print(f"\n⚠️  This will search for and delete all resources matching pattern: {ENDPOINT_NAME}")
        response = input("Do you want to continue? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Cleanup cancelled.")
            sys.exit(0)
        
        # Find and delete related resources
        deleted_resources = find_and_delete_related_resources(sm_client, ENDPOINT_NAME)
    
    # Print summary
    print(f"\n{'='*60}")
    print("✅ Cleanup Complete!")
    print(f"{'='*60}")
    
    print(f"\n📊 Resources Deleted:")
    print(f"  Endpoints: {len(deleted_resources['endpoints'])}")
    for endpoint in deleted_resources['endpoints']:
        print(f"    - {endpoint}")
    
    print(f"  Configurations: {len(deleted_resources['configs'])}")
    for config in deleted_resources['configs']:
        print(f"    - {config}")
    
    print(f"  Models: {len(deleted_resources['models'])}")
    for model in deleted_resources['models']:
        print(f"    - {model}")
    
    total_deleted = (
        len(deleted_resources['endpoints']) + 
        len(deleted_resources['configs']) + 
        len(deleted_resources['models'])
    )
    
    if total_deleted == 0:
        print(f"\n⚠️  No resources were deleted. They may have been already cleaned up.")
    else:
        print(f"\n✓ Total resources deleted: {total_deleted}")
        
        # Remove deployment info file if it exists
        try:
            import os
            if os.path.exists('deployment_info.json'):
                os.remove('deployment_info.json')
                print("✓ Removed deployment_info.json")
        except:
            pass
    
    print(f"\n💡 Note: S3 data and CloudWatch logs are retained for audit purposes.")
    print(f"  You can delete them manually if needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)