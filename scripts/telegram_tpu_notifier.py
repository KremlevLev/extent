from __future__ import annotations

import argparse

from singularity.notifications import (
    TelegramNotifierError,
    accelerator_status_message,
    discover_telegram_chat_ids,
    send_telegram_message,
)


def main(argv: list[str] | None = None, *, enabled: bool = True) -> bool:
    if not enabled:
        print("Telegram notifier is disabled.")
        return False

    parser = argparse.ArgumentParser(description="Report the active JAX accelerator to Telegram.")
    parser.add_argument("--show-chat-ids", action="store_true")
    parser.add_argument("--message", help="Optional extra line for the notification.")
    args = parser.parse_args(argv)

    if args.show_chat_ids:
        try:
            chats = discover_telegram_chat_ids()
        except TelegramNotifierError as exc:
            print(f"Telegram chat discovery failed: {exc}")
            return False
        if not chats:
            print(
                "No chats found. Open the bot in Telegram, press Start, "
                "send a message, and retry."
            )
            return False
        print("Recent Telegram chats:")
        for chat_id, name in chats:
            print(f"  {chat_id}: {name}")
        return True

    try:
        import jax

        text = accelerator_status_message(jax.devices())
    except Exception as exc:
        text = (
            "Singularity runtime status\n"
            "status=JAX initialization failed\n"
            f"error={type(exc).__name__}: {str(exc)[:600]}"
        )
    if args.message:
        text = f"{text}\n{args.message}"
    try:
        message_id = send_telegram_message(text)
    except TelegramNotifierError as exc:
        print(f"Telegram notification failed: {exc}")
        return False
    print(f"Telegram notification sent (message_id={message_id}).")
    return True


if __name__ == "__main__":
    main()
