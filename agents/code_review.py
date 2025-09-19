import os
from typing import Optional
from openai import OpenAI

DEFAULT_SYSTEM = (
    "You are a senior code reviewer. Review the given code for clarity, style, "
    "performance, readability, maintainability, and correctness. "
    "Highlight issues and provide suggestions with examples. "
    "Output sections in markdown:\n"
    "1) Strengths (bullet points)\n"
    "2) Issues/Smells (bullet points)\n"
    "3) Suggested Improvements (bullet points with short code snippets if relevant)\n"
    "4) Overall Rating (1–5 stars, with short justification)."
)

class CodeReviewerAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM,
        temperature: float = 0.3,
        max_tokens: int = 900,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def review(self, code: str, context: Optional[str] = None) -> str:
        """Review code and return markdown feedback."""
        user_content = f"# Code\n```python\n{code.strip()}\n```"
        if context:
            user_content += f"\n\n# Context\n{context}"

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
