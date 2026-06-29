"""本地 vLLM OpenAI-compatible endpoint 客户端。"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "local-mas-model"


class LocalLLMClient:
    """封装 langchain_openai.ChatOpenAI，并提供 OpenAI SDK fallback。"""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._chat = None

    def is_available(self, timeout: float = 3.0) -> tuple[bool, str]:
        """检查 vLLM endpoint 是否可访问。"""
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                data = response.read().decode("utf-8", errors="replace")
            return True, data[:1000]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return False, str(exc)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """调用本地模型。"""
        content, _ = self.invoke_with_metadata(system_prompt, user_prompt, metadata=metadata, max_tokens=max_tokens)
        return content

    def invoke_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """调用本地模型，并返回 OpenAI-compatible 响应元数据。"""
        http_error: Exception | None = None
        try:
            return self._invoke_openai_http_with_metadata(system_prompt, user_prompt, metadata=metadata, max_tokens=max_tokens)
        except Exception as exc:
            http_error = exc
        try:
            return self._invoke_langchain(system_prompt, user_prompt, max_tokens=max_tokens), {}
        except Exception as langchain_exc:
            try:
                return self._invoke_openai_sdk(system_prompt, user_prompt, max_tokens=max_tokens), {}
            except Exception as openai_exc:
                raise RuntimeError(
                    "本地 vLLM 调用失败。"
                    f" urllib fallback 错误: {http_error};"
                    f" langchain_openai 错误: {langchain_exc};"
                    f" openai SDK fallback 错误: {openai_exc}"
                ) from openai_exc

    def invoke_streaming_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call a streaming OpenAI-compatible endpoint and time output chunks.

        vLLM emits Server-Sent Events for chat streaming. We timestamp every
        non-empty content/reasoning delta; these chunk timestamps are the best
        externally observable proxy for token-level timing without modifying
        vLLM internals.
        """
        self._ensure_local_no_proxy()
        request_start = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if seed is not None:
            payload["seed"] = int(seed)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if metadata:
            request_id = metadata.get("X-Request-Id") or metadata.get("request_id_for_backend")
            if request_id:
                headers["X-Request-Id"] = str(request_id)
            headers.update(
                {
                    "X-MAS-Agent-ID": str(metadata.get("agent_id", "")),
                    "X-MAS-Shared-Context-Hash": str(metadata.get("shared_context_hash", "")),
                    "X-MAS-Priority": str(metadata.get("priority", "")),
                }
            )
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        chunks: list[str] = []
        chunk_timestamps: list[float] = []
        response_id = None
        response_model = None
        finish_reason = None
        usage: dict[str, Any] = {}
        system_fingerprint = None
        with opener.open(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_text = line.split("data:", 1)[1].strip()
                if data_text == "[DONE]":
                    break
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError:
                    continue
                response_id = data.get("id") or response_id
                response_model = data.get("model") or response_model
                system_fingerprint = data.get("system_fingerprint") or system_fingerprint
                usage = data.get("usage") or usage
                for choice in data.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if not piece:
                        continue
                    chunks.append(str(piece))
                    chunk_timestamps.append(time.perf_counter())
        end = time.perf_counter()
        intervals = [
            (chunk_timestamps[i] - chunk_timestamps[i - 1]) * 1000.0
            for i in range(1, len(chunk_timestamps))
        ]
        first_token_ms = None
        if chunk_timestamps:
            first_token_ms = (chunk_timestamps[0] - request_start) * 1000.0
        return "".join(chunks), {
            "response_id": response_id,
            "model": response_model,
            "finish_reason": finish_reason,
            "usage": usage,
            "system_fingerprint": system_fingerprint,
            "stream_chunk_count": len(chunk_timestamps),
            "stream_first_token_ms": first_token_ms,
            "stream_chunk_timestamps_perf": chunk_timestamps,
            "stream_inter_token_ms": intervals,
            "stream_e2e_ms": (end - request_start) * 1000.0,
            "timing_source": "openai_compatible_streaming_sse",
        }

    def _invoke_langchain(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        self._ensure_local_no_proxy()
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        result = self._chat.invoke([("system", system_prompt), ("human", user_prompt)])
        return str(result.content)

    def _invoke_openai_sdk(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        self._ensure_local_no_proxy()
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        result = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = result.choices[0].message
        return message.content or str(getattr(message, "reasoning", "") or "")

    def _invoke_openai_http(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        content, _ = self._invoke_openai_http_with_metadata(system_prompt, user_prompt, metadata=metadata, max_tokens=max_tokens)
        return content

    def _invoke_openai_http_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """使用标准库直连 OpenAI-compatible chat completions。"""
        self._ensure_local_no_proxy()
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if metadata:
            request_id = metadata.get("X-Request-Id") or metadata.get("request_id_for_backend")
            if request_id:
                headers["X-Request-Id"] = str(request_id)
            headers.update(
                {
                    "X-MAS-Agent-ID": str(metadata.get("agent_id", "")),
                    "X-MAS-Shared-Context-Hash": str(metadata.get("shared_context_hash", "")),
                    "X-MAS-Priority": str(metadata.get("priority", "")),
                }
            )
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        message = data["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning") or json.dumps(message, ensure_ascii=False)
        return content, {
            "response_id": data.get("id"),
            "model": data.get("model"),
            "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
            "usage": data.get("usage") or {},
            "system_fingerprint": data.get("system_fingerprint"),
        }

    def _ensure_local_no_proxy(self) -> None:
        """本地 endpoint 调用不应经过外部 HTTP proxy。"""
        if "127.0.0.1" in self.base_url or "localhost" in self.base_url:
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ):
                os.environ.pop(key, None)
            existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
            parts = {part.strip() for part in existing.split(",") if part.strip()}
            parts.update({"127.0.0.1", "localhost"})
            value = ",".join(sorted(parts))
            os.environ["NO_PROXY"] = value
            os.environ["no_proxy"] = value


def to_json_prompt(payload: dict[str, Any]) -> str:
    """把节点输入组织成中文 JSON 文本，便于模型稳定读取。"""
    return json.dumps(payload, ensure_ascii=False, indent=2)
