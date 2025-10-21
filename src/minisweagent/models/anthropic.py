import os

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.key_per_thread import get_key_per_thread


class AnthropicModel(LitellmModel):
    """For the use of anthropic models, we need to add explicit cache control marks
    to the messages or we lose out on the benefits of the cache.
    Because break points are limited per key, we also need to rotate between different keys
    if running with multiple agents in parallel threads.
    """

    def __init__(self, **kwargs):
        # Clean up model_kwargs for Anthropic-specific requirements
        if "model_kwargs" in kwargs:
            kwargs = kwargs.copy()
            kwargs["model_kwargs"] = kwargs["model_kwargs"].copy()
            
            # Remove use_cache - handled by set_cache_control, not LiteLLM
            kwargs["model_kwargs"].pop("use_cache", None)
            
            # Force temperature=1 when thinking is enabled
            thinking_config = kwargs["model_kwargs"].get("thinking")
            if thinking_config and thinking_config.get("type") == "enabled":
                kwargs["model_kwargs"]["temperature"] = 1
        
        super().__init__(**kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        api_key = None
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            api_key = get_key_per_thread(rotating_keys.split("::"))
        
        # Force temperature=1 for thinking-enabled models (override any runtime temperature)
        thinking_config = self.config.model_kwargs.get("thinking")
        if thinking_config and thinking_config.get("type") == "enabled":
            kwargs = kwargs.copy() if kwargs else {}
            kwargs["temperature"] = 1
        
        return super().query(set_cache_control(messages), api_key=api_key, **kwargs)
