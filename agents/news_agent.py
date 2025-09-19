# web_agent.py
# Fresh-news agent with tool-calling (+ forced search fallback) using OpenAI SDK.
import os, json, re, html, time
from typing import Dict, Any
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI

# A reasonable User-Agent so some sites don't block us outright
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------- Low-level helpers (no model involved) ----------
def _resp_ok(r: requests.Response) -> None:
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if not any(t in ctype for t in ("text", "xml", "html")):
        raise RuntimeError(f"Unsupported content-type: {ctype}")

def news_search(query: str, num: int = 8) -> Dict[str, Any]:
    """
    Search Google News RSS for latest articles about `query`.
    Returns: {"results": [{"title","url","source","published"}, ...]}
    """
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
    _resp_ok(r)
    soup = BeautifulSoup(r.text, "xml")
    items = []
    for item in soup.find_all("item")[:num]:
        link = item.link.text if item.link else None
        # Some entries wrap real URL in &url= param — unwrap it.
        if link and "url=" in link:
            m = re.search(r"[?&]url=([^&]+)", link)
            if m:
                link = requests.utils.unquote(m.group(1))
        items.append({
            "title": html.unescape(item.title.text if item.title else ""),
            "url": link,
            "source": (item.find("source").text if item.find("source") else ""),
            "published": (item.pubDate.text if item.pubDate else ""),
        })
    return {"results": items}

def fetch_url(url: str) -> Dict[str, Any]:
    """
    Fetch a news article URL and return cleaned text using readability-lxml.
    Returns: {"title": str, "content": str}
    """
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    _resp_ok(r)
    doc = Document(r.text)
    title = html.unescape(doc.short_title() or "")
    article_html = doc.summary()
    soup = BeautifulSoup(article_html, "lxml")
    text = "\n".join(
        p.get_text(" ", strip=True)
        for p in soup.find_all(["p", "li"])
        if p.get_text(strip=True)
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Safety cap to keep tokens reasonable
    return {"title": title, "content": text[:20000]}

# ---------- The agent that lets the model call the tools ----------
class WebAgent:
    """
    Answers current-events questions by calling:
      - news_search(query)
      - fetch_url(url)

    The model orchestrates the calls and writes a concise, cited summary.
    Includes a forced-search fallback so you always get fresh info.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "news_search",
                    "description": "Search recent news for a query and return a list of candidate articles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num": {"type": "integer", "minimum": 1, "maximum": 20}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url",
                    "description": "Fetch and extract readable text from a given news article URL.",
                    "parameters": {
                        "type": "object",
                        "properties": { "url": {"type": "string"} },
                        "required": ["url"]
                    }
                }
            }
        ]

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "news_search":
            # gentle rate-limit
            time.sleep(0.3)
            return news_search(**args)
        if name == "fetch_url":
            time.sleep(0.3)
            return fetch_url(**args)
        return {"error": f"Unknown tool {name}"}

    def ask(self, question: str) -> str:
        """
        Ask about a current topic; the model will search & fetch as needed.
        If the model doesn't call tools on the first round, we force a news_search().
        """
        messages = [
            {"role": "system", "content": (
                "You are a news assistant. ALWAYS use the tools to get fresh info before answering. "
                "After reading 2–4 articles, produce a brief, neutral summary with 2–4 bullets. "
                "Include dates and key figures if present. End with a 'Sources' section listing the URLs used."
            )},
            {"role": "user", "content": question}
        ]

        forced_search_done = False

        for round_idx in range(6):  # allow a few tool/think cycles
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                # Execute tool calls and feed results back
                for call in msg.tool_calls:
                    args = json.loads(call.function.arguments or "{}")
                    result = self._call_tool(call.function.name, args)
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result)
                    })
                continue

            # No tool calls this round
            if round_idx == 0 and not forced_search_done:
                # Force at least one search using the user's question as query
                try:
                    forced = news_search(question, num=8)
                    fake_call_id = "forced-news-search"
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": fake_call_id,
                            "type": "function",
                            "function": {"name": "news_search", "arguments": json.dumps({"query": question, "num": 8})}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": fake_call_id,
                        "name": "news_search",
                        "content": json.dumps(forced)
                    })
                    forced_search_done = True
                    continue
                except Exception as e:
                    return f"Search failed: {type(e).__name__}: {e}"

            # Model produced final text
            return (msg.content or "").strip()

        return "I hit the tool-call limit. Try refining your query."
    

# ---------- Tiny CLI ----------
if __name__ == "__main__":
    agent = WebAgent()
    try:
        q = input("Ask about current news: ").strip()
    except (EOFError, KeyboardInterrupt):
        q = ""
    if not q:
        q = "What are the latest developments in global markets today?"
    print(agent.ask(q))
