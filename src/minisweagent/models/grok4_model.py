"""Grok-4 model adapter that flattens multi-turn conversations.

Grok-4 doesn't support multi-turn conversations in the standard format,
so we flatten user/assistant message pairs into a single user message
with USER: / ASSISTANT: markers.
"""

from minisweagent.models.litellm_model import LitellmModel


def flatten_conversation(messages: list[dict]) -> list[dict]:
    """Flatten multi-turn conversation into single user message format.
    
    Converts user/assistant pairs into single message with USER:/ASSISTANT: markers,
    preserving system messages separately.
    """
    if not messages:
        return messages
    
    # Separate system from conversation messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs = [m for m in messages if m["role"] != "system"]
    
    # No flattening needed for single message
    if len(conv_msgs) <= 1:
        return messages
    
    # Extract text content, handling both string and list formats
    def get_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return content[0].get("text", "") if content else ""
        return str(content)
    
    # Build flattened conversation with USER:/ASSISTANT: markers
    flattened = "\n\n".join(f"{m['role'].upper()}: {get_text(m['content'])}" for m in conv_msgs)
    
    return system_msgs + [{"role": "user", "content": flattened}]


class Grok4Model(LitellmModel):
    """Grok-4 model that flattens multi-turn conversations.
    
    Grok-4 doesn't properly support multi-turn conversations with separate
    user/assistant messages. This adapter flattens the conversation history
    into a single user message with USER:/ASSISTANT: markers.
    
    Example:
        model = Grok4Model(model_name="xai/grok-4")
        response = model.query([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What is 3+3?"}
        ])
        
        # Internally transformed to:
        # [
        #   {"role": "system", "content": "You are a helpful assistant."},
        #   {"role": "user", "content": "USER: What is 2+2?\n\nASSISTANT: 4\n\nUSER: What is 3+3?"}
        # ]
    """
    
    def query(self, messages: list[dict], **kwargs) -> dict:
        """Query Grok-4 with flattened conversation format."""
        flattened_messages = flatten_conversation(messages)
        return super().query(flattened_messages, **kwargs)

