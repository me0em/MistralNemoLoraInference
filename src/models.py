""" Pydantic descriptions in this module has
been writen based on `response_schema.txt` file
"""
from pydantic import BaseModel, Field
from typing import List, Optional


"""
All classes below for handle requests
"""


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float


"""
All classes below for give responses
"""


class CompletionTokensDetails(BaseModel):
    audio_tokens: Optional[int] = None
    reasoning_tokens: int = 0


class PromptTokensDetails(BaseModel):
    audio_tokens: Optional[int] = None
    cached_tokens: int = 0


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails


class ChatCompletionMessage(BaseModel):
    content: str
    refusal: Optional[str] = None
    role: str
    audio: Optional[str] = None
    function_call: Optional[str] = None
    tool_calls: Optional[str] = None


class Choice(BaseModel):
    finish_reason: str
    index: int
    logprobs: Optional[str] = None
    message: ChatCompletionMessage


class ChatCompletion(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: CompletionUsage
