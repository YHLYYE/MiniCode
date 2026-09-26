"""Web fetch and search tools"""
import json
import os
import re
import urllib.request

from core.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web via Tavily (set TAVILY_API_KEY in .env)."""

    name = "WebSearch"
    description = (
        "Search the web and return results with title, URL, and content. "
        "Use to look up errors, docs, or current information."
    )
    input_schema = {
        "query": {
            "type": "string",
            "description": "Search query",
        },
    }
    is_readonly = True
    is_concurrency_safe = True

    async def execute(self, query: str) -> str:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return (
                "WebSearch not configured: set TAVILY_API_KEY in .env "
                "(get a free key at https://tavily.com)."
            )
        try:
            payload = json.dumps({
                "api_key": api_key,
                "query": query,
                "max_results": 5,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            results = data.get("results", [])[:5]
            if not results:
                return f"No results found for: {query}"
            return "\n\n".join(
                f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}"
                for r in results
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
