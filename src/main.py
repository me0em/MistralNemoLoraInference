from fastapi import FastAPI, HTTPException, Header
import uvicorn

from models import (
    Message,
    CompletionRequest,
    CompletionTokensDetails,
    PromptTokensDetails,
    CompletionUsage,
    ChatCompletionMessage,
    Choice,
    ChatCompletion
)

from ai import (
    model,
    tokenizer,
    generate_text
)


def build_response(raw_answer: str, model: str) -> ChatCompletion:
    """ Get raw string-like model response, model name and return
    it like OpenAI Server do
    """
    output = ChatCompletion(
        id="chatcmpl-Random00000000000000000000000",
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=raw_answer,
                    role="assistant"
                )
            )
        ],
        created=1730550198,
        model=model,
        object='chat.completion',
        usage=CompletionUsage(
            completion_tokens=7,
            prompt_tokens=13,
            total_tokens=20,
            completion_tokens_details=CompletionTokensDetails(
                audio_tokens=None,
                reasoning_tokens=0
            ),
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None,
                cached_tokens=0
            )
        )
    )

    return output


app = FastAPI()


@app.post("/chat/completions", response_model=ChatCompletion)
async def chat_completions(request: CompletionRequest,
                           authorization: str = Header(None)):
    """ This endpoint emulates I/O of
    real OpenAI Completions API
    """
    print(f"New request: {request}")

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        query=request.messages[0].content
    )

    response = build_response(
        raw_answer=output,
        model=request.model
    )

    return response


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="debug",
        workers=1
    )
