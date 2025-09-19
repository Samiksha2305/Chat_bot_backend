# router_agent.py
from openai import OpenAI
from backend.agents.sql_agent import SQLAgent
from backend.agents.analyst_agent import AnalystAgent
from backend.agents.notes_agent import NotesAgent
from backend.agents.code_review import CodeReviewerAgent
from backend.agents.news_agent import WebAgent
from backend.agents.chatbot_agent import ChatbotAgent

class RouterAgent:
    def __init__(self):
        self.client = OpenAI()
        self.sql = SQLAgent()
        self.analyst = AnalystAgent()
        self.notes = NotesAgent()
        self.code = CodeReviewerAgent()
        self.news = WebAgent()
        self.chat = ChatbotAgent()

    def classify(self, msg: str) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify intent: chatbot, sql, analyst, notes, code_review, news"},
                {"role": "user", "content": msg},
            ],
        )
        return resp.choices[0].message.content.strip().lower()

    def run(self, msg: str) -> str:
        intent = self.classify(msg)
        if "sql" in intent:
            return self.sql.run(msg)
        elif "analyst" in intent:
            return self.analyst.run(msg)
        elif "notes" in intent:
            return self.notes.summarize(msg)
        elif "code" in intent:
            return self.code.review(msg)
        elif "news" in intent:
            return self.news.ask(msg)
        else:
            return self.chat.run(msg)
