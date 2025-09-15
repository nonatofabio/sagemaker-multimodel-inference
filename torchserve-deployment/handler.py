"""
TorchServe handler for PII masking and prompt injection detection models.
"""
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class MultiModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.pii_model = None
        self.pii_tokenizer = None
        self.injection_model = None
        self.injection_tokenizer = None
        self.device = None

    def initialize(self, context):
        """Initialize models and tokenizers."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load PII masking model
        pii_model_path = context.system_properties.get("model_dir") + "/pii_model"
        self.pii_tokenizer = AutoTokenizer.from_pretrained(pii_model_path)
        self.pii_model = AutoModelForTokenClassification.from_pretrained(pii_model_path)
        self.pii_model.to(self.device)
        self.pii_model.eval()
        
        # Load prompt injection model
        injection_model_path = context.system_properties.get("model_dir") + "/injection_model"
        self.injection_tokenizer = AutoTokenizer.from_pretrained(injection_model_path)
        self.injection_model = AutoModelForSequenceClassification.from_pretrained(injection_model_path)
        self.injection_model.to(self.device)
        self.injection_model.eval()

    def preprocess(self, data):
        """Preprocess input data."""
        inputs = []
        for row in data:
            body = row.get("body") or row.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = body.decode('utf-8')
            if isinstance(body, str):
                body = json.loads(body)
            inputs.append(body)
        return inputs

    def inference(self, inputs):
        """Run inference on preprocessed data."""
        results = []
        
        for input_data in inputs:
            text = input_data.get("text", "")
            task = input_data.get("task", "pii_masking")
            
            if task == "pii_masking":
                result = self._mask_pii(text)
            elif task == "prompt_injection":
                result = self._detect_injection(text)
            else:
                result = {"error": f"Unknown task: {task}"}
            
            results.append(result)
        
        return results

    def _mask_pii(self, text):
        """Mask PII in text."""
        inputs = self.pii_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.pii_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)
        
        tokens = self.pii_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.pii_model.config.id2label[id.item()] for id in predicted_token_class_ids[0]]
        
        masked_text = self._apply_masking(tokens, labels, text)
        
        return {
            "masked_text": masked_text,
            "entities": self._extract_entities(tokens, labels)
        }

    def _detect_injection(self, text):
        """Detect prompt injection."""
        inputs = self.injection_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.injection_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
        label = self.injection_model.config.id2label[predicted_class_id]
        
        return {
            "is_injection": label == "INJECTION",
            "confidence": confidence,
            "label": label
        }

    def _apply_masking(self, tokens, labels, original_text):
        """Apply PII masking to text."""
        masked_tokens = []
        for token, label in zip(tokens, labels):
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label.split("-")[1]
                masked_tokens.append(f"[{entity_type}]")
            else:
                masked_tokens.append(token)
        
        return self.pii_tokenizer.convert_tokens_to_string(masked_tokens)

    def _extract_entities(self, tokens, labels):
        """Extract detected entities."""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": label.split("-")[1],
                    "start": i,
                    "end": i,
                    "word": token
                }
            elif label.startswith("I-") and current_entity:
                current_entity["end"] = i
                current_entity["word"] += token
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def postprocess(self, inference_output):
        """Postprocess inference results."""
        return inference_output
