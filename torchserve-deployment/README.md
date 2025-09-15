# TorchServe Deployment for SageMaker

This directory contains the TorchServe implementation of the PII masking and prompt injection detection models.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create model archive:**
   ```bash
   python create_model_archive.py
   ```

3. **Deploy to SageMaker:**
   ```bash
   python deploy_torchserve.py
   ```

4. **Test the deployment:**
   ```bash
   python client_torchserve.py
   ```

## Key Differences from HuggingFace Deployment

- **Single Handler**: Both models run in one TorchServe handler
- **Custom Logic**: Direct control over preprocessing/postprocessing
- **Model Archive**: Uses .mar files instead of HuggingFace containers
- **Batch Processing**: Built-in TorchServe batching capabilities

## Files

- `handler.py` - TorchServe custom handler
- `create_model_archive.py` - Creates .mar file
- `deploy_torchserve.py` - Deploys to SageMaker
- `client_torchserve.py` - Client for testing
- `model_config.py` - Configuration settings
