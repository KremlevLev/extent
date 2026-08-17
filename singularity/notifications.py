from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable, Iterable
from typing import Any
from urllib import error, parse, request


class TelegramNotifierError(RuntimeError):
    """A notifier failure that never includes the bot token in its message."""


def load_secret(name: str) -> str:
    """Read a secret from the environment or Kaggle's notebook secret store."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
    except Exception as exc:  # Kaggle wraps disabled/missing secrets in backend-specific errors.
        raise TelegramNotifierError(
            f"Missing {name}. Add it under Kaggle Add-ons > Secrets and enable notebook access."
        ) from exc
    if not value:
        raise TelegramNotifierError(f"Secret {name} is empty.")
    return value


def _telegram_call(
    token: str,
    method: str,
    payload: dict[str, str],
    *,
    timeout: float = 10.0,
    urlopen: Callable[..., Any] = request.urlopen,
) -> dict[str, Any]:
    endpoint = f"https://api.telegram.org/bot{token}/{method}"
    encoded = parse.urlencode(payload).encode("utf-8")
    http_request = request.Request(endpoint, data=encoded, method="POST")
    try:
        with urlopen(http_request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise TelegramNotifierError(f"Telegram returned HTTP {exc.code}.") from exc
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TelegramNotifierError("Telegram request failed.") from exc
    if not result.get("ok"):
        description = str(result.get("description", "unknown Bot API error"))
        raise TelegramNotifierError(f"Telegram rejected the request: {description}")
    return result


def send_telegram_message(
    text: str,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    timeout: float = 10.0,
    urlopen: Callable[..., Any] = request.urlopen,
) -> int:
    """Send a plain-text notification and return Telegram's message id."""
    token = token or load_secret("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or load_secret("TELEGRAM_CHAT_ID")
    result = _telegram_call(
        token,
        "sendMessage",
        {"chat_id": chat_id, "text": text},
        timeout=timeout,
        urlopen=urlopen,
    )
    return int(result["result"]["message_id"])


def discover_telegram_chat_ids(
    *,
    token: str | None = None,
    timeout: float = 10.0,
    urlopen: Callable[..., Any] = request.urlopen,
) -> tuple[tuple[str, str], ...]:
    """Return chat ids/names from recent bot updates without printing the token."""
    token = token or load_secret("TELEGRAM_BOT_TOKEN")
    result = _telegram_call(token, "getUpdates", {}, timeout=timeout, urlopen=urlopen)
    chats: dict[str, str] = {}
    for update in result.get("result", []):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        if "id" not in chat:
            continue
        name = chat.get("title") or chat.get("username") or chat.get("first_name") or "unknown"
        chats[str(chat["id"])] = str(name)
    return tuple(sorted(chats.items()))


def accelerator_status_message(devices: Iterable[Any]) -> str:
    """Describe the accelerator JAX actually initialized without rejecting fallbacks."""
    devices = list(devices)
    if not devices:
        return "Singularity runtime status\nstatus=error\naccelerator=no JAX devices"
    platforms = sorted({str(getattr(device, "platform", "unknown")) for device in devices})
    process_count = len({int(getattr(device, "process_index", 0)) for device in devices})
    accelerator = "+".join(platform.upper() for platform in platforms)
    tpu_active = "tpu" in platforms
    return (
        "Singularity runtime status\n"
        f"status={'TPU ready' if tpu_active else 'TPU not active'}\n"
        f"host={socket.gethostname()}\n"
        f"accelerator={accelerator} devices={len(devices)} processes={process_count}\n"
        "next=run the planned experiment in this notebook process"
    )
