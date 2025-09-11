"""
Configuration file for SageMaker deployment of HuggingFace models.
"""

# AWS Settings
AWS_REGION = "us-east-1"  # Change to your preferred region
EXECUTION_ROLE_NAME = "AmazonSageMaker-ExecutionRole"  # Will be created if doesn't exist

# Endpoint Configuration
ENDPOINT_NAME = "huggingface-multi-model-endpoint"
ENDPOINT_CONFIG_NAME = f"{ENDPOINT_NAME}-config"

# Model Configurations
MODELS_CONFIG = {
    "pii_masking": {
        "name": "pii-masking-model",
        "model_id": "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
        "task": "token-classification",
        "variant_name": "PIIMasking",
        "instance_type": "ml.g5.xlarge",
        "initial_instance_count": 1,
        "initial_variant_weight": 0.5
    },
    "prompt_injection": {
        "name": "prompt-injection-model",
        "model_id": "protectai/deberta-v3-base-prompt-injection-v2",
        "task": "text-classification",
        "variant_name": "PromptInjection",
        "instance_type": "ml.g5.xlarge",
        "initial_instance_count": 1,
        "initial_variant_weight": 0.5
    }
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_input_length": 1024,
    "max_total_tokens": 2048,
    "max_batch_total_tokens": 8192,
    "max_batch_prefill_tokens": 4096
}

# Monitoring and Logging
ENABLE_DATA_CAPTURE = False  # Set to True to enable data capture for monitoring
DATA_CAPTURE_SAMPLING_PERCENTAGE = 10  # Percentage of requests to capture

# Timeouts (in seconds)
DEPLOYMENT_TIMEOUT = 1800  # 30 minutes
INFERENCE_TIMEOUT = 60  # 1 minute per request