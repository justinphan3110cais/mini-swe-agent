"""
Config helper functions to integrate with the models.yaml configuration
similar to how intphys2_eval.py works.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from .custom_llm_agent import CustomLLMAgentModel

logger = logging.getLogger("config_helper")


def get_agent_config(model_alias: str, models_config: str) -> Dict[str, Any]:
    """
    Load agent configuration from models.yaml file.
    This mimics the get_agent_config function from intphys2_eval.py
    
    Args:
        model_alias: The model alias (e.g., 'gpt-5-high', 'grok-4')
        models_config: Path to the models.yaml config file
        
    Returns:
        Dict containing model configuration
    """
    models_config_path = Path(models_config)
    if not models_config_path.exists():
        raise FileNotFoundError(f"Models config file not found: {models_config_path}")
    
    with open(models_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_alias not in config:
        available_models = list(config.keys())
        raise ValueError(f"Model alias '{model_alias}' not found in config. Available models: {available_models}")
    
    model_config = config[model_alias]
    model_name = model_config.get('model')
    generation_config = model_config.get('generation_config', {})
    
    if not model_name:
        raise ValueError(f"Model name not specified for alias '{model_alias}'")
    
    logger.info(f"Loading model '{model_name}' for alias '{model_alias}'")
    logger.info(f"Generation config: {generation_config}")
    
    return {
        'model': model_name,
        'generation_config': generation_config
    }


def get_model_with_yaml_config(model_alias: str, models_config: str) -> CustomLLMAgentModel:
    """
    Create a CustomLLMAgentModel instance from YAML config.
    
    Args:
        model_alias: The model alias (e.g., 'gpt-5-high', 'grok-4')
        models_config: Path to the models.yaml config file
        
    Returns:
        CustomLLMAgentModel instance
    """
    try:
        agent_config = get_agent_config(model_alias, models_config)
        
        return CustomLLMAgentModel(
            model_name=agent_config['model'],
            generation_config=agent_config['generation_config']
        )
        
    except Exception as e:
        logger.error(f"Failed to create model for alias '{model_alias}': {e}")
        raise