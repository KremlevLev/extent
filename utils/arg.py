import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Запуск инференса FlaxQwen на TPU"
    )
    # Добавляем аргумент для изменения количества слоев в конфигурации
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Изменить количество слоев num_hidden_layers в конфигурации модели",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Входной текст (промпт) для генерации",
    )
    parser.add_argument(
        "--send_notification",
        action="store_true",
        help="Отправить уведомление в Telegram при запуске скрипта",
    )

    return parser.parse_args()