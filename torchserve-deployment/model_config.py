"""
Configuration for TorchServe deployment.
"""

# Model configurations
MODELS = {
    "pii_masking": {
        "model_id": "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
        "task": "token-classification"
    },
    "prompt_injection": {
        "model_id": "protectai/deberta-v3-base-prompt-injection-v2", 
        "task": "text-classification"
    }
}

# TorchServe configuration
TORCHSERVE_CONFIG = {
    "model_name": "multi-security-model",
    "handler": "handler.py",
    "runtime": "python",
    "batch_size": 16,
    "max_batch_delay": 100,
    "response_timeout": 120,
    "initial_workers": 1,
    "max_workers": 4
}

# AWS SageMaker configuration
SAGEMAKER_CONFIG = {
    "instance_type": "ml.g5.xlarge",
    "initial_instance_count": 1,
    "model_data_url": None,  # Will be set during deployment
    "framework_version": "2.6.0",
    "py_version": "py312"
}
