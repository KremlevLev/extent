import json

import pytest

from singularity.notifications import (
    TelegramNotifierError,
    discover_telegram_chat_ids,
    send_telegram_message,
    tpu_ready_message,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_send_telegram_message_uses_post_and_returns_message_id():
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["method"] = http_request.get_method()
        captured["body"] = http_request.data
        captured["timeout"] = timeout
        return FakeResponse({"ok": True, "result": {"message_id": 17}})

    message_id = send_telegram_message(
        "TPU ready", token="secret", chat_id="42", timeout=3, urlopen=fake_urlopen
    )
    assert message_id == 17
    assert captured == {"method": "POST", "body": b"chat_id=42&text=TPU+ready", "timeout": 3}


def test_discover_chat_ids_deduplicates_recent_updates():
    def fake_urlopen(*_args, **_kwargs):
        return FakeResponse(
            {
                "ok": True,
                "result": [
                    {"message": {"chat": {"id": 42, "first_name": "Lev"}}},
                    {"message": {"chat": {"id": 42, "first_name": "Lev"}}},
                ],
            }
        )

    assert discover_telegram_chat_ids(token="secret", urlopen=fake_urlopen) == (("42", "Lev"),)


def test_tpu_ready_message_requires_tpu():
    class Device:
        def __init__(self, platform, process_index=0):
            self.platform = platform
            self.process_index = process_index

    message = tpu_ready_message([Device("tpu"), Device("tpu")])
    assert "devices=2" in message
    with pytest.raises(TelegramNotifierError, match="TPU is not ready"):
        tpu_ready_message([Device("cpu")])

