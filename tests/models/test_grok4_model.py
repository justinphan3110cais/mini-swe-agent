"""Tests for Grok4Model conversation flattening."""

import pytest

from minisweagent.models.grok4_model import Grok4Model, flatten_conversation


def test_flatten_conversation_basic():
    """Test basic multi-turn conversation flattening."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+3?"},
    ]
    
    result = flatten_conversation(messages)
    
    # Should have system message + one flattened user message
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helpful assistant."
    
    assert result[1]["role"] == "user"
    expected_content = "USER: What is 2+2?\n\nASSISTANT: 4\n\nUSER: What is 3+3?"
    assert result[1]["content"] == expected_content


def test_flatten_conversation_single_user_message():
    """Test that single user message is not flattened."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    result = flatten_conversation(messages)
    
    # Should remain unchanged
    assert len(result) == 2
    assert result == messages


def test_flatten_conversation_no_system():
    """Test flattening without system message."""
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
    ]
    
    result = flatten_conversation(messages)
    
    # Should have one flattened user message
    assert len(result) == 1
    assert result[0]["role"] == "user"
    expected_content = "USER: Question 1\n\nASSISTANT: Answer 1\n\nUSER: Question 2"
    assert result[0]["content"] == expected_content


def test_flatten_conversation_with_structured_content():
    """Test flattening with structured content (e.g., from cache control)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is 2+2?", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+3?"},
    ]
    
    result = flatten_conversation(messages)
    
    # Should extract text from structured content
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    expected_content = "USER: What is 2+2?\n\nASSISTANT: 4\n\nUSER: What is 3+3?"
    assert result[1]["content"] == expected_content


def test_flatten_conversation_empty():
    """Test empty messages list."""
    result = flatten_conversation([])
    assert result == []


def test_flatten_conversation_long_history():
    """Test flattening a long conversation history."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
        {"role": "user", "content": "Q4"},
    ]
    
    result = flatten_conversation(messages)
    
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    
    expected_content = (
        "USER: Q1\n\n"
        "ASSISTANT: A1\n\n"
        "USER: Q2\n\n"
        "ASSISTANT: A2\n\n"
        "USER: Q3\n\n"
        "ASSISTANT: A3\n\n"
        "USER: Q4"
    )
    assert result[1]["content"] == expected_content


def test_grok4_model_class_exists():
    """Test that Grok4Model can be instantiated."""
    model = Grok4Model(model_name="xai/grok-4")
    assert model is not None
    assert model.config.model_name == "xai/grok-4"


def test_flatten_conversation_multiple_systems():
    """Test handling multiple system messages."""
    messages = [
        {"role": "system", "content": "System 1"},
        {"role": "system", "content": "System 2"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "Another user message"},
    ]
    
    result = flatten_conversation(messages)
    
    # All system messages should be preserved
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "System 1"
    assert result[1]["role"] == "system"
    assert result[1]["content"] == "System 2"
    
    # Conversation should be flattened
    assert result[2]["role"] == "user"
    expected_content = (
        "USER: User message\n\n"
        "ASSISTANT: Assistant message\n\n"
        "USER: Another user message"
    )
    assert result[2]["content"] == expected_content

