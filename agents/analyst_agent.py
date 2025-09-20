import os
from openai import OpenAI


class AnalystAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.last_plot_path = None

    def run(self, user_message: str) -> str:
        """Simplified analyst that provides text-based data insights"""
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Analyze any CSV data provided and give insights. Since plotting is not available, describe what charts would be useful."},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or "Analysis complete."
        except Exception as e:
            return f"Analysis error: {str(e)}"
