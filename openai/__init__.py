from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict

import requests


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


class _ChatCompletions:
    def __init__(self, client: "OpenAI") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> SimpleNamespace:
        return self._client._post("/chat/completions", kwargs)


class _Chat:
    def __init__(self, client: "OpenAI") -> None:
        self.completions = _ChatCompletions(client)


class OpenAI:
    def __init__(self, *, api_key: str | None = None, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat = _Chat(self)

    def _post(self, path: str, payload: Dict[str, Any]) -> SimpleNamespace:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return _to_namespace(response.json())

