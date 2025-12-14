import json
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger_from_config(
        config_path: Optional[str] = None,
        log_dir: str = ".",
        log_file: str = "app.log",
        for_child_process: bool = False
) -> None:
    """
    Настройка логгера из JSON конфигурации.

    Args:
        config_path: Путь к JSON конфигурации. Если None, используется встроенная конфигурация.
        log_dir: Директория для лог-файлов
        log_file: Имя лог-файла
        for_child_process: Флаг для подпроцессов multiprocessing
    """
    # Создаем директорию для логов
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Загружаем конфигурацию
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = get_default_config()

    # Настраиваем пути в конфигурации
    update_config_paths(config, log_path, for_child_process)

    # Применяем конфигурацию
    logging.config.dictConfig(config)

    if for_child_process:
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.removeHandler(handler)


def get_default_config() -> Dict[str, Any]:
    """Возвращает конфигурацию по умолчанию."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - PID=%(process)d - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s: %(message)s"
            }
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "app.log",
                "mode": "a",
                "encoding": "utf-8"
            },
            "console_handler": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["file_handler", "console_handler"],
                "propagate": False
            }
        }
    }


def update_config_paths(
        config: Dict[str, Any],
        log_path: str,
        for_child_process: bool = False
) -> None:
    """Обновляет пути в конфигурации."""
    # Обновляем путь к файлу
    if "handlers" in config and "file_handler" in config["handlers"]:
        config["handlers"]["file_handler"]["filename"] = log_path

    # Для подпроцессов используем только файловый хендлер
    if for_child_process:
        if "loggers" in config and "" in config["loggers"]:
            config["loggers"][""]["handlers"] = ["file_handler"]


def setup_simple_logger(
        level: int = logging.INFO,
        format_str: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> None:
    """Простая настройка логгера без конфигурационного файла."""
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# Сохраняем обратную совместимость
def setup_logger(
        log_dir: str = ".",
        log_file: str = "app.log",
        for_child_process: bool = False
) -> None:
    """Старая функция для обратной совместимости."""
    setup_logger_from_config(
        config_path=None,
        log_dir=log_dir,
        log_file=log_file,
        for_child_process=for_child_process
    )