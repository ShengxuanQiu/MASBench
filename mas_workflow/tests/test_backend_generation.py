from app.llm_backends import OpenAICompatibleLLM


def test_openai_backend_applies_resolved_generation_contract(monkeypatch):
    backend = OpenAICompatibleLLM(
        model="m",
        base_url="http://localhost/v1",
        generation={
            "max_tokens": 48,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    captured = {}

    def invoke(system, user, *, metadata, max_tokens):
        captured.update(metadata)
        return "ok", {"completion_ts": 1, "first_token_ts": 1, "usage": {}}

    monkeypatch.setattr(backend.client, "invoke_with_metadata", invoke)
    backend.invoke("system", "user", {})
    assert backend.max_tokens == 48
    assert backend.client.temperature == 0
    assert captured["_request_payload"] == {
        "temperature": 0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_fixed_replay_payload_is_not_overridden(monkeypatch):
    backend = OpenAICompatibleLLM(
        model="m",
        base_url="http://localhost/v1",
        generation={"temperature": 0, "seed": 42},
    )
    captured = {}

    def invoke(system, user, *, metadata, max_tokens):
        captured.update(metadata)
        return "ok", {"completion_ts": 1, "first_token_ts": 1, "usage": {}}

    monkeypatch.setattr(backend.client, "invoke_streaming_with_metadata", invoke)
    backend.invoke("system", "user", {
        "_fixed_payload": True,
        "_request_payload": {"temperature": 0.7, "seed": 9},
    })
    assert captured["_request_payload"] == {"temperature": 0.7, "seed": 9}
