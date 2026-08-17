import json
import sys
from types import SimpleNamespace

from scripts import telegram_tpu_notifier
from singularity.notifications import (
    accelerator_status_message,
    discover_telegram_chat_ids,
    send_telegram_message,
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


def test_accelerator_status_reports_tpu_and_gpu():
    class Device:
        def __init__(self, platform, process_index=0):
            self.platform = platform
            self.process_index = process_index

    tpu_message = accelerator_status_message([Device("tpu"), Device("tpu")])
    gpu_message = accelerator_status_message([Device("gpu"), Device("gpu")])
    assert "status=TPU ready" in tpu_message and "devices=2" in tpu_message
    assert "status=TPU not active" in gpu_message and "accelerator=GPU" in gpu_message


def test_notifier_can_be_disabled_without_checking_runtime(capsys):
    assert not telegram_tpu_notifier.main(["--not-a-real-option"], enabled=False)
    assert "disabled" in capsys.readouterr().out


def test_notifier_sends_gpu_status_and_jax_errors(monkeypatch):
    class Device:
        platform = "gpu"
        process_index = 0

    sent = []
    monkeypatch.setattr(telegram_tpu_notifier, "send_telegram_message", lambda text: sent.append(text) or 1)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: [Device()]))
    assert telegram_tpu_notifier.main([])
    assert "TPU not active" in sent.pop()

    def fail_devices():
        raise RuntimeError("accelerator unavailable")

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=fail_devices))
    assert telegram_tpu_notifier.main([])
    assert "JAX initialization failed" in sent.pop()
