# notes_agent.py
import os
import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI


DEFAULT_SYSTEM = (
    "You are an executive note-taker. Convert raw, messy meeting notes into a clear, "
    "concise brief with these sections in order:\n"
    "1) Summary (4–7 bullet points)\n"
    "2) Decisions (bullet list)\n"
    "3) Action Items (markdown table with columns: Owner | Task | Due | Priority | Status)\n"
    "4) Deadlines (bullet list of key dates)\n"
    "5) Risks/Blockers (bullet list)\n"
    "6) Open Questions (bullet list)\n"
    "Write tersely. Only use information explicitly found in the notes. "
    "If a section has no info, write '—'. Never invent names, dates, or tasks."
)

TEMPLATE = """\
# Input Notes
{notes}

# Context
- Audience: {audience}
- Tone: {tone}
- Max length: {max_len} words (soft cap)

# Output Rules
- Use markdown. No preambles, no apologies.
- Convert dates to ISO-like format (e.g., 2025-09-20) when possible.
- Normalize people to Firstname Lastname if available; otherwise keep as given.
- Action Items table MUST have header row exactly: Owner | Task | Due | Priority | Status
"""

class NotesAgent:
    """
    Notes summarization agent.
    - Summarizes messy notes into a standard markdown brief.
    - Allows light knobs (audience/tone/max_len).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM,
        temperature: float = 0.2,
        max_tokens: int = 900
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def summarize(
        self,
        notes: str,
        audience: str = "Team",
        tone: str = "crisp, neutral, professional",
        max_len: int = 350,
    ) -> str:
        """
        Turn raw notes into a structured markdown brief.
        """
        user_content = TEMPLATE.format(
            notes=notes.strip(),
            audience=audience,
            tone=tone,
            max_len=max_len,
        )

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


# ---------- Tiny CLI ----------
if __name__ == "__main__":
    print("NotesAgent: paste notes, then Ctrl-D/Ctrl-Z when done.\n")
    try:
        buf = []
        while True:
            line = input()
            buf.append(line)
    except (EOFError, KeyboardInterrupt):
        pass

    raw = "\n".join(buf).strip() or """
- kickoff for Q4 growth push; target +15% MRR by 2025-10-31
- web perf: LCP too high on checkout (4.8s). Need fix before BF/CM
- Priya: own LCP investigation; sync with infra
- Ken: draft pricing experiment; ETA next Friday
- Decision: move ad spend from Meta to Search (trial 2 weeks)
- Risk: data pipeline flaky on weekends; on-call saturation
"""
    agent = NotesAgent()
    print("\n--- Summary ---\n")
    print(agent.summarize(raw, audience="Leads", tone="concise, executive", max_len=280))
