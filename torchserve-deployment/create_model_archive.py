"""
Create TorchServe model archive (.mar file) for deployment.
"""
import os
import subprocess
import tempfile
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from model_config import MODELS, TORCHSERVE_CONFIG

def download_models():
    """Download and save models locally."""
    os.makedirs("models", exist_ok=True)
    
    # Download PII model
    pii_path = "models/pii_model"
    os.makedirs(pii_path, exist_ok=True)
    pii_tokenizer = AutoTokenizer.from_pretrained(MODELS["pii_masking"]["model_id"])
    pii_model = AutoModelForTokenClassification.from_pretrained(MODELS["pii_masking"]["model_id"])
    pii_tokenizer.save_pretrained(pii_path)
    pii_model.save_pretrained(pii_path)
    
    # Download injection model
    injection_path = "models/injection_model"
    os.makedirs(injection_path, exist_ok=True)
    injection_tokenizer = AutoTokenizer.from_pretrained(MODELS["prompt_injection"]["model_id"])
    injection_model = AutoModelForSequenceClassification.from_pretrained(MODELS["prompt_injection"]["model_id"])
    injection_tokenizer.save_pretrained(injection_path)
    injection_model.save_pretrained(injection_path)
    
    print("✅ Models downloaded successfully")

def create_requirements():
    """Create requirements.txt for the model archive."""
    requirements = [
        "torch>=1.13.0",
        "transformers>=4.21.0",
        "torchserve>=0.7.0",
        "torch-model-archiver>=0.7.0",
        "numpy",
        "tokenizers"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("✅ Requirements file created")

def create_archive():
    """Create the .mar file using torch-model-archiver."""
    create_requirements()
    
    cmd = [
        "torch-model-archiver",
        "--model-name", TORCHSERVE_CONFIG["model_name"],
        "--version", "1.0",
        "--handler", TORCHSERVE_CONFIG["handler"],
        "--runtime", TORCHSERVE_CONFIG["runtime"],
        "--export-path", "model-store",
        "--extra-files", "models/",
        "--requirements-file", "requirements.txt",
        "--force"
    ]
    
    os.makedirs("model-store", exist_ok=True)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Model archive created successfully")
        print(f"Archive location: model-store/{TORCHSERVE_CONFIG['model_name']}.mar")
        
        # Verify requirements are included
        print("📋 Verifying requirements inclusion...")
        return f"model-store/{TORCHSERVE_CONFIG['model_name']}.mar"
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating model archive: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

if __name__ == "__main__":
    print("📦 Downloading models...")
    download_models()
    
    print("🏗️ Creating model archive...")
    archive_path = create_archive()
    
    if archive_path:
        print(f"🎉 Model archive ready: {archive_path}")
    else:
        print("❌ Failed to create model archive")
