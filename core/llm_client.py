from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests
import yaml


class LLMClient:
    def __init__(self) -> None:
        self._load_dotenv()
        self.config = self._load_config()
        self.provider = self.config.get("llm", {}).get("provider", "groq")
        self.groq_model = self.config.get("llm", {}).get("groq", {}).get("model", "llama-3.1-8b-instant")
        self.groq_api_key = os.getenv("GROQ_API_KEY") or self.config.get("llm", {}).get("groq", {}).get("api_key", "")
        self.ollama_model = self.config.get("llm", {}).get("ollama", {}).get("model", "llama3.2")
        self.ollama_base_url = self.config.get("llm", {}).get("ollama", {}).get("base_url", "http://localhost:11434")

    def _load_dotenv(self) -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _resolve_env_vars(self, value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: os.getenv(match.group(1), ""), value)
        if isinstance(value, dict):
            return {key: self._resolve_env_vars(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        return value

    def _load_config(self) -> dict[str, Any]:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        return self._resolve_env_vars(raw_config)

    def get_runtime_settings(self) -> tuple[str, str, str]:
        provider = self.provider
        model = self.groq_model if provider == "groq" else self.ollama_model
        return provider, model, self.groq_api_key

    def get_available_models(self, provider: str) -> list[str]:
        if provider == "ollama":
            return ["llama3.2", "llama3.1", "phi3"]
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

    def generate(self, prompt: str) -> str:
        provider, model, api_key = self.get_runtime_settings()
        if provider == "ollama":
            return self._generate_with_ollama(prompt, model)
        return self._generate_with_groq(prompt, model, api_key)

    def _generate_with_groq(self, prompt: str, model: str, api_key: str) -> str:
        if not api_key:
            return "Groq API key is not configured."
        try:
            from groq import Groq
        except ImportError:
            return "groq package is not installed."

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _generate_with_ollama(self, prompt: str, model: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception:
            return "Ollama service is not available."


llm_client = LLMClient()
