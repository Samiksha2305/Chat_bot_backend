# chatbot_agent.py
import os
import time
from typing import List, Dict, Optional
from openai import OpenAI

class ChatbotAgent:
    """
    Minimal conversational agent with memory.
    - Keeps message history so the model has context across turns.
    - Light history pruning to avoid token bloat.
    - Simple retry on transient errors.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system: str = "You are a friendly, concise assistant.",
        temperature: float = 0.7,
        max_turns: int = 4,              # keep last N user/assistant turns (not counting system)
        max_retries: int = 2,             # small retry budget for transient failures
        retry_backoff_sec: float = 1.0,   # backoff base
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

        # Conversation memory: system + alternating user/assistant messages
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system}
        ]

    # ---------- core chat method ----------
    def run(self, user_message: str) -> str:
        """
        Send one user turn, return assistant reply. History is preserved.
        """
        self.messages.append({"role": "user", "content": user_message})
        self._prune_history()

        # tiny retry loop
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                )
                assistant_text = (resp.choices[0].message.content or "").strip()
                self.messages.append({"role": "assistant", "content": assistant_text})
                return assistant_text
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (2 ** attempt))
                    continue
                raise

    # ---------- utilities ----------
    def reset(self, new_system: Optional[str] = None):
        """
        Clear history. Optionally set a fresh system prompt.
        """
        sys_prompt = new_system if new_system is not None else self.messages[0]["content"]
        self.messages = [{"role": "system", "content": sys_prompt}]

    def set_system(self, new_system: str):
        """
        Update system prompt while keeping history (applies to future turns).
        """
        self.messages[0]["content"] = new_system

    def set_model(self, model: str):
        self.model = model

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def get_history(self) -> List[Dict[str, str]]:
        """
        Read-only peek at the chat log (useful for debugging).
        """
        return list(self.messages)

    # ---------- internals ----------
    def _prune_history(self):
        """
        Keep only the last `max_turns` user/assistant exchanges to control token growth.
        System message is always kept at index 0.
        """
        # messages[1:] are alternating user/assistant turns
        if self.max_turns is None or self.max_turns <= 0:
            return
        # Each turn is (user, assistant). Keep last 2*max_turns entries (plus system).
        excess = len(self.messages) - (1 + 2 * self.max_turns)
        if excess > 0:
            # drop from just after system message
            self.messages[1:1 + excess] = []

# ---------- run as a tiny terminal chatbot ----------
if __name__ == "__main__":
    print("ChatbotAgent REPL. Type 'reset' to clear memory, 'exit' to quit.\n")
    bot = ChatbotAgent()

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            if user.lower().startswith("reset"):
                # allow new system prompt after 'reset: <prompt>'
                _, _, maybe_system = user.partition(":")
                bot.reset(new_system=maybe_system.strip() or None)
                print("Bot: (context cleared)\n")
                continue

            reply = bot.run(user)
            print(f"Bot: {reply}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nGoodbye!")
