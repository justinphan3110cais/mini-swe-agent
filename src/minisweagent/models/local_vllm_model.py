"""Local vLLM model using OpenAI client with custom base URL."""
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("local_vllm_model")


@dataclass
class LocalVllmModelConfig:
    model_name: str
    api_base_url: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Allow env variable to override
        if self.api_base_url is None:
            self.api_base_url = os.getenv("OPENAI_BASE_URL", "http://0.0.0.0:8000/v1")

class LocalVllmModel:
    """Model class for locally hosted vLLM servers using OpenAI-compatible API."""
    
    def __init__(self, **kwargs):
        self.config = LocalVllmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        # Strip "local/" prefix if present for API calls
        self.actual_model_name = self.config.model_name
        if self.actual_model_name.startswith("local/"):
            self.actual_model_name = self.actual_model_name[6:]  # Remove "local/" prefix
        
        # Initialize OpenAI client with custom base URL
        self.client = openai.OpenAI(
            base_url=self.config.api_base_url,
            timeout=1200.0,
        )
        
        logger.info(f"Initialized LocalVllmModel with model={self.config.model_name} (api_model={self.actual_model_name}), base_url={self.config.api_base_url}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                openai.NotFoundError,
                openai.PermissionDeniedError,
                openai.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Filter out litellm-specific and config parameters that OpenAI client doesn't understand
        # These are either used during client initialization or are litellm-specific
        params_to_filter = {
            'drop_params',        # litellm parameter
            'custom_llm_provider', # litellm parameter
            'api_base',           # litellm uses this instead of base_url
            'api_base_url',       # Our config parameter, already used in client init
            'api_key',            # Already used in client init
        }
        
        # Merge model_kwargs and kwargs, filtering out non-OpenAI params
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k not in params_to_filter}
        
        try:
            return self.client.chat.completions.create(
                model=self.actual_model_name,  # Use the model name without "local/" prefix
                messages=messages,
                **filtered_kwargs
            )
        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}. Check OPENAI_API_KEY or api_key setting.")
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        
        # Since it's a local model, cost is 0 (or you could implement token-based pricing)
        cost = 0.0
        
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        return {
            "content": response.choices[0].message.content or "",
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

