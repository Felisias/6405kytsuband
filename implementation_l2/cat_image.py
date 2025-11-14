import requests
import numpy as np
import cv2
from abc import ABC, abstractmethod
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple
import os

from implementation_self import ImageProcessingSelf
from implementation import ImageProcessing


class CatImage(ABC):
    """
    Абстрактный базовый класс для работы с изображениями животных.
    Реализует основные операции: загрузка, обработка, сохранение.
    """

    def __init__(self, image_url: str, breed: str):
        """
        Инициализация объекта изображения.

        Args:
            image_url (str): URL для загрузки изображения
            breed (str): Порода животного
        """
        self._image_url = image_url
        self._breed = breed
        self._image: Optional[np.ndarray] = None
        self._processed_cv2_image: Optional[np.ndarray] = None
        self._processed_self_image: Optional[np.ndarray] = None

    @property
    def breed(self) -> str:
        """Property для получения породы животного."""
        return self._breed

    @property
    def image(self) -> Optional[np.ndarray]:
        """Property для получения исходного изображения."""
        return self._image

    @property
    def processed_cv2_image(self) -> Optional[np.ndarray]:
        """Property для получения изображения, обработанного OpenCV."""
        return self._processed_cv2_image

    @property
    def processed_self_image(self) -> Optional[np.ndarray]:
        """Property для получения изображения, обработанного пользовательским методом."""
        return self._processed_self_image

    @abstractmethod
    def download_image(self) -> None:
        """Абстрактный метод для загрузки изображения."""
        pass

    @abstractmethod
    def process_image(self) -> None:
        """Абстрактный метод для обработки изображения."""
        pass

    @abstractmethod
    def save_image(self, save_dir: str, index: int) -> None:
        """Абстрактный метод для сохранения изображений."""
        pass

    def __add__(self, other: 'CatImage') -> 'CatImage':
        """
        Перегрузка оператора сложения.
        Складывает два изображения (поэлементное сложение).

        Args:
            other (CatImage): Другое изображение для сложения

        Returns:
            CatImage: Новый объект с результатом сложения
        """
        if self._image is None or other._image is None:
            raise ValueError("Оба изображения должны быть загружены для сложения")

        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры для сложения")

        result = self._create_copy()

        img1_float = self._image.astype(np.float32)
        img2_float = other._image.astype(np.float32)

        result_float = img1_float * 0.8 + img2_float * 0.2

        max_val = np.max(result_float)

        result_normalized = (result_float / max_val) * 255.0

        result._image = result_normalized.astype(np.uint8)

        return result

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        """
        Перегрузка оператора вычитания.
        Вычитает одно изображение из другого.

        Args:
            other (CatImage): Изображение для вычитания

        Returns:
            CatImage: Новый объект с результатом вычитания
        """
        if self._image is None or other._image is None:
            raise ValueError("Оба изображения должны быть загружены для вычитания")

        if self._image.shape != other._image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры для вычитания")

        result = self._create_copy()

        img1_float = self._image.astype(np.float32)
        img2_float = other._image.astype(np.float32)

        result_float = np.abs(img1_float - img2_float)

        min_val = np.min(result_float)

        result_float = (result_float - min_val)
        max_val = np.max(result_float)


        result_normalized = (result_float / max_val) * 255.0

        result._image = result_normalized.astype(np.uint8)

        return result

    def __str__(self) -> str:
        """
        Перегрузка преобразования в строку.

        Returns:
            str: Строковое представление объекта
        """
        return f"CatImage(breed='{self._breed}', url='{self._image_url}')"

    def _create_copy(self) -> 'CatImage':
        """Создает копию объекта с теми же параметрами."""
        # Этот метод будет переопределен в дочерних классах
        raise NotImplementedError("Должен быть реализован в дочерних классах")

    @staticmethod
    def convert_to_numpy(image_bytes: bytes) -> np.ndarray:
        """
        Статический метод для преобразования байтов в numpy-массив.

        Args:
            image_bytes (bytes): Байты изображения

        Returns:
            np.ndarray: Изображение в виде numpy массива
        """
        img = Image.open(BytesIO(image_bytes))

        # Конвертируем в RGB если нужно
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return np.array(img)


class CatImageColoredPlusGray(CatImage):
    """
    Класс для работы с цветными изображениями.
    Наследует абстрактный класс CatImage.
    """

    def download_image(self) -> None:
        """
        Загружает изображение по URL и преобразует в numpy-массив.
        """
        response = requests.get(self._image_url)

        if response.status_code == 200:
            self._image = self.convert_to_numpy(response.content)
        else:
            raise Exception(f"Ошибка загрузки изображения: {response.status_code}")

    def process_image(self) -> None:
        """
        Обрабатывает изображение методами выделения контуров.
        Использует как OpenCV, так и пользовательскую реализацию.
        """
        if self._image is None:
            raise ValueError("Изображение не загружено.")

        # Метод 1: Выделение контуров с помощью OpenCV
        processor = ImageProcessing()
        self._processed_cv2_image = processor.edge_detection(self._image)

        # Метод 2: Пользовательская реализация выделения контуров
        processor_self = ImageProcessingSelf()
        self._processed_self_image = processor_self.edge_detection(self._image)

    def save_image(self, save_dir: str, index: int) -> None:
        """
        Сохраняет исходное и обработанные изображения.

        Args:
            save_dir (str): Директория для сохранения
            index (int): Порядковый номер изображения
        """

        # Используем os.path.join
        if self.image is not None:
            original_path = os.path.join(save_dir, f"{index}_{self.breed}_original.png")
            cv2.imwrite(original_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

        if self.processed_cv2_image is not None:
            processed_cv2_path = os.path.join(save_dir, f"{index}_{self.breed}_processed_cv2.png")
            cv2.imwrite(processed_cv2_path, self.processed_cv2_image)

        if self.processed_self_image is not None:
            processed_self_path = os.path.join(save_dir, f"{index}_{self.breed}_processed_self.png")
            cv2.imwrite(processed_self_path, self.processed_self_image)

    def _create_copy(self) -> 'CatImageColoredPlusGray':
        """Создает копию объекта CatImageColored."""
        copy_obj = CatImageColoredPlusGray(self._image_url, self.breed)
        copy_obj._image = self.image.copy() if self.image is not None else None
        return copy_obj
