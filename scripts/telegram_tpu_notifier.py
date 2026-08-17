from __future__ import annotations

import argparse

from singularity.notifications import (
    TelegramNotifierError,
    discover_telegram_chat_ids,
    send_telegram_message,
    tpu_ready_message,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Notify Telegram when this JAX process sees TPU.")
    parser.add_argument("--show-chat-ids", action="store_true")
    parser.add_argument("--message", help="Optional extra line for the notification.")
    args = parser.parse_args(argv)

    if args.show_chat_ids:
        chats = discover_telegram_chat_ids()
        if not chats:
            raise TelegramNotifierError(
                "No chats found. Open the bot in Telegram, press Start, send a message, and retry."
            )
        print("Recent Telegram chats:")
        for chat_id, name in chats:
            print(f"  {chat_id}: {name}")
        return

    import jax

    text = tpu_ready_message(jax.devices())
    if args.message:
        text = f"{text}\n{args.message}"
    message_id = send_telegram_message(text)
    print(f"Telegram notification sent (message_id={message_id}).")


if __name__ == "__main__":
    main()
