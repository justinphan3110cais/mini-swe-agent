import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import asyncio

# Import llm_agents from the local copy
from .llm_agents import get_llm_agent_class

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("custom_llm_agent")


@dataclass
class CustomLLMAgentConfig:
    model_name: str
    generation_config: dict = field(default_factory=dict)
    max_retries: int = 3
    retry_delay: float = 1.0
    models_config_path: str = ""


class CustomLLMAgentModel:
    def __init__(self, **kwargs):
        self.config = CustomLLMAgentConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        # If models_config_path is provided, load from YAML config
        if self.config.models_config_path:
            from .config_helper import get_agent_config
            agent_config = get_agent_config(self.config.model_name, self.config.models_config_path)
            model_name = agent_config['model']
            generation_config = agent_config['generation_config']
            logger.info(f"Loaded config from YAML for alias '{self.config.model_name}' -> model '{model_name}'")
        else:
            # Fallback to direct configuration
            model_name = self.config.model_name
            generation_config = self.config.generation_config
        
        # Initialize the LLM agent using your llm_agents.py
        self.agent = get_llm_agent_class(
            model=model_name,
            generation_config=generation_config
        )
        
        logger.info(f"Initialized CustomLLMAgentModel with model: {model_name}")

    def _query_with_retry(self, messages: list[dict[str, str]], **kwargs):
        """Query with retry logic similar to your llm_agents implementation."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Use the agent's completions method
                response = self.agent.completions(messages, **kwargs)
                
                # Update cost tracking
                if response.token_usage:
                    # Estimate cost based on token usage (you can adjust these rates)
                    estimated_cost = (
                        response.token_usage.input_tokens * 0.000001 +  # $1 per 1M input tokens
                        response.token_usage.output_tokens * 0.000003   # $3 per 1M output tokens
                    )
                    self.cost += estimated_cost
                    GLOBAL_MODEL_STATS.add(estimated_cost)
                
                self.n_calls += 1
                return response.content or ""
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed for model {self.config.model_name}: {e}"
                )
                
                if attempt < self.config.max_retries - 1:
                    import time
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed for model {self.config.model_name}")
                    raise last_exception

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Main query method that mini-swe-agent expects."""
        try:
            content = self._query_with_retry(messages, **kwargs)
            return {"content": content}
        except Exception as e:
            logger.error(f"Query failed after all retries: {e}")
            # Return empty content to allow the agent to continue
            return {"content": f"Error: {str(e)}"}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}