from typing import Protocol

class OpenAIRequestTool(Protocol):
    def request_openai_completion(self, prompt: str, model: str, max_tokens: int) -> str:
        ...

    def o1_preview_simulated_request(self, system_prompt: str, user_prompt: str) -> str:
        ...

    async def o1_preview_simulated_request_async(self, system_prompt: str, user_prompt: str) -> str:
        ...
    