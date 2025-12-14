"""
async_processing - Пакет для асинхронной обработки изображений с котами
"""

from async_processing.async_processing_lib.implementation import CatImageAbstract, CatImageRGB, CatImageGrayscale
from async_processing.async_processing_lib.implementation import CatImageProcessor
from async_processing.async_processing_lib.logging_config import setup_logger_from_config, setup_logger

__all__ = [
    "CatImageAbstract",
    "CatImageRGB",
    "CatImageGrayscale",
    "CatImageProcessor",
    "setup_logger_from_config",
    "setup_logger",
]