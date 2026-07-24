import json
import requests
import os
from kaggle_secrets import UserSecretsClient
# Пытаемся прочитать переменные из окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
# Локально в VS Code подгружаем .env, если библиотека установлена
try:
    user_secrets = UserSecretsClient()
    TELEGRAM_BOT_TOKEN = user_secrets.get_secret("TELEGRAM_BOT_TOKEN")
    TELEGRAM_USER_ID = user_secrets.get_secret("TELEGRAM_USER_ID")
except Exception:
    # Откат на os.getenv для локального запуска в VS Code
    import os
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

def send_telegram_notification(text: str):
    """Отправляет текстовое уведомление в Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_USER_ID, "text": text, "parse_mode": "Markdown"}

    payload = {
        "chat_id": TELEGRAM_USER_ID, 
        "text": text, 
        "parse_mode": "Markdown"
    }
    
    # Отправляем POST-запрос на эндпоинт Telegram
    response = requests.post(url, json=payload)
    return response.json()

def send_tg():
    send_telegram_notification('success')