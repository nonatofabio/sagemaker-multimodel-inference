"""
SageMaker Client Utility for HuggingFace Models.

This module provides easy-to-use functions for invoking the deployed
PII masking and prompt injection detection models on SageMaker.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from functools import wraps
import boto3
from botocore.exceptions import ClientError
import os

from config import (
    ENDPOINT_NAME,
    MODELS_CONFIG,
    AWS_REGION,
    INFERENCE_TIMEOUT
)

def get_actual_endpoint_name():
    """
    Get the actual endpoint name from deployment_info.json if it exists,
    otherwise fall back to the config value.
    """
    deployment_info_path = os.path.join(os.path.dirname(__file__), 'deployment_info.json')
    if os.path.exists(deployment_info_path):
        try:
            with open(deployment_info_path, 'r') as f:
                deployment_info = json.load(f)
                actual_name = deployment_info.get('endpoint_name')
                if actual_name:
                    return actual_name
        except Exception as e:
            pass  # Fall back to default
    return ENDPOINT_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions on transient errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplicative factor for delay increase
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code in ['ThrottlingException', 'ServiceUnavailable', 'TooManyRequestsException']:
                        last_error = e
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed with {error_code}. Retrying in {current_delay} seconds...")
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            raise
                    else:
                        raise
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            if last_error:
                raise last_error
                
        return wrapper
    return decorator


class SageMakerClient:
    """Client for interacting with SageMaker multi-variant endpoint."""
    
    def __init__(self, endpoint_name: str = None, region: str = AWS_REGION):
        """
        Initialize the SageMaker client.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint (if None, reads from deployment_info.json)
            region: AWS region where the endpoint is deployed
        """
        # Use provided endpoint name, or try to get from deployment_info.json, or fall back to config
        self.endpoint_name = endpoint_name if endpoint_name else get_actual_endpoint_name()
        self.region = region
        self.runtime_client = boto3.client(
            'sagemaker-runtime',
            region_name=region
        )
        self.pii_variant = MODELS_CONFIG["pii_masking"]["variant_name"]
        self.injection_variant = MODELS_CONFIG["prompt_injection"]["variant_name"]
    
    @retry_on_error(max_retries=3)
    def _invoke_endpoint(
        self,
        payload: Dict[str, Any],
        target_variant: str,
        timeout: int = INFERENCE_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Internal method to invoke SageMaker endpoint.
        
        Args:
            payload: Input data to send to the model
            target_variant: Name of the model variant to invoke
            timeout: Request timeout in seconds
        
        Returns:
            Response from the model as a dictionary
        """
        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=json.dumps(payload),
                TargetVariant=target_variant
            )
            
            result = json.loads(response['Body'].read().decode())
            return result
            
        except ClientError as e:
            logger.error(f"Error invoking endpoint: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def mask_pii(
        self,
        text: Union[str, List[str]],
        return_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Mask PII in the given text using the PII masking model.
        
        Args:
            text: Single text string or list of texts to process
            return_entities: Whether to return detected entities
        
        Returns:
            Dictionary containing masked text and optionally detected entities
        """
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        payload = {
            "inputs": texts
        }
        
        start_time = time.time()
        response = self._invoke_endpoint(payload, self.pii_variant)
        latency = time.time() - start_time
        
        # Process response
        results = []
        for idx, result in enumerate(response):
            masked_text = self._apply_pii_masking(texts[idx], result)
            
            processed_result = {
                "original_text": texts[idx],
                "masked_text": masked_text,
                "latency_seconds": latency / len(texts)
            }
            
            if return_entities:
                entities = self._extract_entities(result)
                processed_result["entities"] = entities
            
            results.append(processed_result)
        
        return results[0] if single_input else results
    
    def detect_prompt_injection(
        self,
        text: Union[str, List[str]],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect prompt injection in the given text.
        
        Args:
            text: Single text string or list of texts to analyze
            threshold: Confidence threshold for injection detection
        
        Returns:
            Dictionary containing injection detection results
        """
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        payload = {
            "inputs": texts
        }
        
        start_time = time.time()
        response = self._invoke_endpoint(payload, self.injection_variant)
        latency = time.time() - start_time
        
        # Process response
        results = []
        
        # Handle different response formats for text-classification models
        # The model returns a flat list for both single and batch inputs
        # Single input: [{"label": "SAFE", "score": 0.99}]
        # Batch input: [{"label": "SAFE", "score": 0.99}, {"label": "INJECTION", "score": 0.95}, ...]
        
        # For single input, wrap in list to normalize processing
        if single_input:
            # Response is [{"label": "SAFE", "score": 0.99}]
            # We need to process just this one result
            processed_response = [response]  # Wrap for consistent processing
        else:
            # For batch input, each item in response corresponds to one input text
            # But we need to wrap each in a list for consistent processing
            processed_response = [[item] for item in response]
        
        for idx, result in enumerate(processed_response):
            # Extract scores and determine if injection detected
            injection_score = 0.0
            safe_score = 0.0
            
            # result is now always a list of dict items
            for label_score in result:
                if isinstance(label_score, dict) and 'label' in label_score:
                    label = label_score['label'].upper()
                    if label in ['INJECTION', 'MALICIOUS', 'UNSAFE']:
                        injection_score = label_score['score']
                    elif label in ['SAFE', 'BENIGN', 'CLEAN']:
                        safe_score = label_score['score']
            
            is_injection = injection_score > threshold
            
            processed_result = {
                "text": texts[idx],
                "is_injection": is_injection,
                "injection_score": injection_score,
                "safe_score": safe_score,
                "confidence": max(injection_score, safe_score),
                "classification": "INJECTION_DETECTED" if is_injection else "SAFE",
                "latency_seconds": latency / len(texts)
            }
            
            results.append(processed_result)
        
        return results[0] if single_input else results
    
    def batch_process(
        self,
        texts: List[str],
        operation: str,
        batch_size: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batches.
        
        Args:
            texts: List of texts to process
            operation: Either 'mask_pii' or 'detect_injection'
            batch_size: Number of texts to process in each batch
            **kwargs: Additional arguments for the operation
        
        Returns:
            List of results for all texts
        """
        if operation not in ['mask_pii', 'detect_injection']:
            raise ValueError("Operation must be 'mask_pii' or 'detect_injection'")
        
        operation_func = getattr(self, operation)
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} of {(len(texts) - 1) // batch_size + 1}")
            
            try:
                batch_results = operation_func(batch, **kwargs)
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Add error results for failed batch
                for text in batch:
                    all_results.append({
                        "text": text,
                        "error": str(e),
                        "status": "failed"
                    })
        
        return all_results
    
    def _apply_pii_masking(self, text: str, entities: List[Dict]) -> str:
        """
        Apply masking to text based on detected entities.
        
        Args:
            text: Original text
            entities: List of detected entities with labels and positions
        
        Returns:
            Text with PII masked
        """
        # Sort entities by start position in reverse order to maintain positions
        sorted_entities = sorted(
            entities,
            key=lambda x: x.get('start', 0),
            reverse=True
        )
        
        masked_text = text
        for entity in sorted_entities:
            if entity.get('entity_group') or entity.get('entity'):
                label = entity.get('entity_group', entity.get('entity', 'PII'))
                label = label.replace('B-', '').replace('I-', '').upper()
                
                start = entity.get('start', 0)
                end = entity.get('end', len(text))
                
                # Create mask based on entity type
                mask = f"[{label}]"
                
                # Apply mask
                masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text
    
    def _extract_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Extract and format detected entities.
        
        Args:
            entities: Raw entities from model
        
        Returns:
            Formatted list of entities
        """
        formatted_entities = []
        
        for entity in entities:
            if entity.get('score', 0) > 0.5:  # Filter low-confidence detections
                formatted_entity = {
                    "type": entity.get('entity_group', entity.get('entity', 'UNKNOWN')).replace('B-', '').replace('I-', ''),
                    "text": entity.get('word', ''),
                    "start": entity.get('start', 0),
                    "end": entity.get('end', 0),
                    "confidence": entity.get('score', 0.0)
                }
                formatted_entities.append(formatted_entity)
        
        return formatted_entities
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of both model variants.
        
        Returns:
            Dictionary with health status for each variant
        """
        health_status = {
            "endpoint": self.endpoint_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variants": {}
        }
        
        # Test PII masking variant
        try:
            test_text = "Test health check"
            result = self.mask_pii(test_text, return_entities=False)
            health_status["variants"][self.pii_variant] = {
                "status": "healthy",
                "response_time": result.get("latency_seconds", 0)
            }
        except Exception as e:
            health_status["variants"][self.pii_variant] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Test prompt injection variant
        try:
            test_text = "Test health check"
            result = self.detect_prompt_injection(test_text)
            health_status["variants"][self.injection_variant] = {
                "status": "healthy",
                "response_time": result.get("latency_seconds", 0)
            }
        except Exception as e:
            health_status["variants"][self.injection_variant] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall status
        all_healthy = all(
            v.get("status") == "healthy"
            for v in health_status["variants"].values()
        )
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        
        return health_status


# Convenience functions for direct usage
def create_client(endpoint_name: str = None, region: str = AWS_REGION) -> SageMakerClient:
    """Create and return a SageMaker client instance."""
    return SageMakerClient(endpoint_name, region)


def mask_pii_text(text: str, client: Optional[SageMakerClient] = None) -> str:
    """
    Convenience function to mask PII in text.
    
    Args:
        text: Text to mask
        client: Optional client instance (creates new if not provided)
    
    Returns:
        Masked text string
    """
    if client is None:
        client = create_client()
    
    result = client.mask_pii(text)
    return result["masked_text"]


def check_prompt_injection(text: str, client: Optional[SageMakerClient] = None) -> bool:
    """
    Convenience function to check for prompt injection.
    
    Args:
        text: Text to check
        client: Optional client instance (creates new if not provided)
    
    Returns:
        True if injection detected, False otherwise
    """
    if client is None:
        client = create_client()
    
    result = client.detect_prompt_injection(text)
    return result["is_injection"]