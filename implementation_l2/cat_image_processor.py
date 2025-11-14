import time
import os
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv

from implementation_l2 import CatImageColoredPlusGray

# Загрузка переменных окружения
load_dotenv()


class CatImageProcessor:
    """
    Класс для управления процессом загрузки и обработки изображений.
    Инкапсулирует работу с API и координацию обработки.
    """

    def __init__(self, api_key: str, api_url: str, limit: int = 5):
        """
        Инициализация процессора изображений.

        Args:
            api_key (str): API ключ для доступа к сервису
            limit (int): Количество изображений для загрузки
        """
        self._api_key = api_key
        self._limit = limit
        self._base_url = api_url
        self._images_data: List[Dict[str, Any]] = []

    @property
    def limit(self) -> int:
        """Property для получения лимита изображений."""
        return self._limit

    @staticmethod
    def measure_time(func):
        """
        Статический метод-декоратор для измерения времени выполнения.

        Args:
            func: Функция для обертывания

        Returns:
            Обернутая функция с измерением времени
        """

        def wrapper(self, *args, **kwargs):
            print(f" Начало выполнения {func.__name__}...")
            start_time = time.time()

            result = func(self, *args, **kwargs)

            end_time = time.time()
            execution_time = end_time - start_time
            print(f" Завершено {func.__name__}. Время: {execution_time:.4f} секунд")

            return result

        return wrapper

    @measure_time
    def fetch_images(self) -> List[Dict[str, Any]]:
        """
        Получает данные об изображениях с API.

        Returns:
            List[Dict]: Список данных об изображениях
        """
        headers = {'x-api-key': self._api_key}
        params = {'limit': self.limit, 'has_breeds': True}  # Только изображения с информацией о породе

        response = requests.get(self._base_url, headers=headers, params=params)

        if response.status_code == 200:
            self._images_data = response.json()
            return self._images_data
        else:
            error_msg = f" Ошибка загрузки: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)

    @measure_time
    def process_images(self) -> None:
        """
        Основной метод обработки изображений.
        Координирует загрузку, обработку и сохранение.
        """

        # Получаем данные об изображениях
        images_data = self.fetch_images()

        if not images_data:
            print(" Нет данных для обработки")
            return

        # Создаем директорию для сохранения
        save_dir = 'images'
        os.makedirs(save_dir, exist_ok=True)

        # Обрабатываем каждое изображение
        successful_processed = 0

        for index, data in enumerate(images_data):
            try:
                print(f"\n--- Обработка изображения {index + 1}/{len(images_data)} ---")

                image_url = data['url']
                breed = data['breeds'][0]['name'] if data.get('breeds') else 'unknown_breed'

                print(f" Порода: {breed}")
                print(f" URL: {image_url}")

                # Создаем объект

                cat_image = CatImageColoredPlusGray(image_url, breed)

                # Загружаем и обрабатываем изображение
                cat_image.download_image()
                cat_image.process_image()

                # Сохраняем изображения
                cat_image.save_image(save_dir, index + 1)

                successful_processed += 1

            except Exception as e:
                print(f" Ошибка при обработке изображения {index + 1}: {str(e)}")
                continue

        print(f"\n Обработка завершена! Успешно обработано: {successful_processed}/{len(images_data)}")

