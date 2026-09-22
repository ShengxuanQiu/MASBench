import json

from app.semantic.canonicalize import TransformersTokenizer
from app.semantic.model import OperationRecord
from app.semantic.replay import VLLMOpenAIBackend


def test_transformers_v5_batch_encoding_extracts_integer_ids():
    adapter = object.__new__(TransformersTokenizer)

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [[11, 12, 13]], "attention_mask": [[1, 1, 1]]}

    adapter._tokenizer = FakeTokenizer()
    assert adapter.encode_request({"messages": []}) == [11, 12, 13]


def test_vllm_length_lock_removes_stream_options(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "id": "response",
                "usage": {"completion_tokens": 4},
                "choices": [{"message": {"content": "done"}}],
            }).encode()

    class Opener:
        def open(self, request, timeout):
            captured.update(json.loads(request.data))
            return Response()

    monkeypatch.setattr("app.semantic.replay.urllib.request.build_opener", lambda *args: Opener())
    operation = OperationRecord(
        operation_id="op", operation_type="llm", task_id="task", stage_instance_id="stage",
        session_id="session", llm={"recorded_output_length": 4},
    )
    result = VLLMOpenAIBackend("http://localhost/v1").generate(
        {"model": "m", "messages": [], "stream": True,
         "stream_options": {"include_usage": True}}, operation, "length_locked")
    assert result.output_length == 4
    assert captured["stream"] is False
    assert "stream_options" not in captured
    assert captured["min_tokens"] == captured["max_tokens"] == 4
