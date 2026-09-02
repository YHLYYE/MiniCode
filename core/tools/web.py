"""Web fetch and search tools"""
import json
import re
import urllib.request
import urllib.parse

from core.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key required)"""

    name = "WebSearch"
    description = "Search the web and return results"
    input_schema = {
        "query": {"type": "string", "description": "Search query"},
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, query: str) -> str:
        try:
            url = (
                "https://api.duckduckgo.com/?"
                f"q={urllib.parse.quote(query)}&format=json"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "MiniCode/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            results = data.get("RelatedTopics", [])[:5]
            if not results:
                return f"No results found for: {query}"
            return "\n".join(
                f"- {r.get('Text', '')}"
                for r in results
                if r.get("Text")
            )
        except Exception as e:
            return f"WebSearch error: {e}"


class WebFetchTool(Tool):
    """Fetch a URL and extract text content"""

    name = "WebFetch"
    description = "Fetch a URL and return its text content"
    input_schema = {
        "url": {"type": "string", "description": "URL to fetch"},
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, url: str) -> str:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "MiniCode/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            # Strip scripts and styles
            content = re.sub(
                r'<script[^>]*>.*?</script>', '',
                content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(
                r'<style[^>]*>.*?</style>', '',
                content, flags=re.DOTALL | re.IGNORECASE
            )
            # Strip HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            # Normalize whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)

            return content[:5000]
        except Exception as e:
            return f"WebFetch error: {e}"
