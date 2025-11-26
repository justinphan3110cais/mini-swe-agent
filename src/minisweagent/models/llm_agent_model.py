"""LLM Agent Model using llm_agents.py for all model providers."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.llm_agents import get_llm_agent_class

logger = logging.getLogger("llm_agent_model")


@dataclass
class LLMAgentModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class LLMAgentModel:
    """Model class that uses llm_agents.py for all providers."""
    
    def __init__(self, **kwargs):
        self.config = LLMAgentModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        # Extract generation_config from model_kwargs if present
        generation_config = self.config.model_kwargs.copy()
        
        # Initialize the llm_agent using get_llm_agent_class
        self.agent = get_llm_agent_class(
            model=self.config.model_name,
            generation_config=generation_config
        )
        
        logger.info(f"Initialized LLMAgentModel with model={self.config.model_name}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt,)),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Query the model with retry logic."""
        # Use the completions method from llm_agents
        response = self.agent.completions(messages, **kwargs)
        return response

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the model and return response in mini-swe-agent format."""
        response = self._query(messages, **kwargs)
        
        # Extract cost and token usage from llm_agents response
        cost = response.token_usage.cost if response.token_usage else 0.0
        content = response.content or ""
        
        # Update counters
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        return {
            "content": content,
            "extra": {
                "response": response.raw if response.raw else {},
                "token_usage": {
                    "input_tokens": response.token_usage.input_tokens if response.token_usage else 0,
                    "output_tokens": response.token_usage.output_tokens if response.token_usage else 0,
                    "cached_tokens": response.token_usage.cached_tokens if response.token_usage else 0,
                    "total_tokens": response.token_usage.total_tokens if response.token_usage else 0,
                    "cost": cost,
                }
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for agent prompts."""
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

