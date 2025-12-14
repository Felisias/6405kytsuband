"""
Модуль реализации обработки изображений
"""

from .cat_image import CatImageAbstract, CatImageRGB, CatImageGrayscale
from .cat_image_processor import CatImageProcessor

__all__ = [
    "CatImageAbstract",
    "CatImageRGB",
    "CatImageGrayscale",
    "CatImageProcessor",
]