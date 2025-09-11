import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from .custom_llm_agent import CustomLLMAgentModel

logger = logging.getLogger("yaml_model_loader")


def load_models_config(config_path: str) -> Dict[str, Any]:
    """Load models configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Models config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_from_yaml_config(model_alias: str, config_path: str) -> CustomLLMAgentModel:
    """
    Create a model instance from YAML config, similar to how intphys2_eval.py works.
    
    Args:
        model_alias: The model alias (e.g., 'gpt-5-high', 'grok-4')
        config_path: Path to the models.yaml config file
        
    Returns:
        CustomLLMAgentModel instance configured with the specified model
    """
    try:
        models_config = load_models_config(config_path)
        
        if model_alias not in models_config:
            raise ValueError(f"Model alias '{model_alias}' not found in config. Available models: {list(models_config.keys())}")
        
        model_config = models_config[model_alias]
        model_name = model_config.get('model')
        generation_config = model_config.get('generation_config', {})
        
        if not model_name:
            raise ValueError(f"Model name not specified for alias '{model_alias}'")
        
        logger.info(f"Loading model '{model_name}' for alias '{model_alias}'")
        logger.info(f"Generation config: {generation_config}")
        
        # Create the model instance
        return CustomLLMAgentModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
    except Exception as e:
        logger.error(f"Failed to load model '{model_alias}' from config: {e}")
        raise