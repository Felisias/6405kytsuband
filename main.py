import os
import sys
import argparse
import cv2
import numpy as np
from implementation_l2 import CatImageProcessor, CatImageColoredPlusGray


def  process_from_api(limit=1):
    """
    Обрабатывает изображения через API.
    """
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    if not api_key:
        print("Ошибка: не найден API_KEY")
        return
    if not api_url:
        print("Ошибка: не найден API_URL")
        return

    processor = CatImageProcessor(api_key=api_key, api_url=api_url, limit=limit)
    processor.process_images()

def process_from_folder(operation="add"):
    """
    Обрабатывает изображения из папки images.

    Args:
        operation (str): "add" для сложения, "sub" для вычитания
    """

    images_dir = "images"

    if not os.path.exists(images_dir):
        print(f" Папка {images_dir} не существует")
        return

    # Ищем все оригинальные изображения
    original_files = [f for f in os.listdir(images_dir) if f.endswith('_original.png')]

    if not original_files:
        print(f" В папке {images_dir} нет оригинальных изображений")
        return

    print(f" Найдено {len(original_files)} изображений для обработки")

    for original_file in original_files:
        try:
            # Извлекаем префикс
            prefix = original_file.replace('_original.png', '')

            # Формируем пути к файлам
            original_path = os.path.join(images_dir, original_file)
            processed_path = os.path.join(images_dir, f"{prefix}_processed_self.png")

            if not os.path.exists(processed_path):
                print(f"  Файл {processed_path} не найден, пропускаем")
                continue

            # Загружаем изображения как numpy массив
            original_img = cv2.imread(original_path)
            processed_img = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

            if original_img is None or processed_img is None:
                print(f" Ошибка загрузки изображений для {prefix}")
                continue

            # Преобразуем контуры в 3-канальное изображение
            processed_3d = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

            # Выполняем операцию
            # Выполняем операцию
            if operation == "add":
                # Создаем временные объекты CatImage
                temp_original = CatImageColoredPlusGray("", "temp")
                temp_processed = CatImageColoredPlusGray("", "temp")

                # Записываем в них наши изображения
                temp_original._image = original_img
                temp_processed._image = processed_3d

                # Используем перегруженный +
                result_obj = temp_original + temp_processed
                result = result_obj.image

                output_path = os.path.join(images_dir, f"{prefix}_addition.png")
                print(f" Сложение: {original_file} + контуры")
            else:
                # Создаем временные объекты CatImage
                temp_original = CatImageColoredPlusGray("", "temp")
                temp_processed = CatImageColoredPlusGray("", "temp")

                # Записываем в них наши изображения
                temp_original._image = original_img
                temp_processed._image = processed_3d

                # Используем перегруженный -
                result_obj = temp_original - temp_processed
                result = result_obj.image

                output_path = os.path.join(images_dir, f"{prefix}_subtraction.png")
                print(f" Вычитание: {original_file} - контуры")

            # Сохраняем результат
            cv2.imwrite(output_path, result)

        except Exception as e:
            print(f" Ошибка при обработке {original_file}: {str(e)}")


def main():
    """
    Основная функция программы.
    Поддерживает два режима работы через аргументы командной строки.
    """

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Обработка изображений собак')
    parser.add_argument('--mode', choices=['api', 'folder'], required=True,
                        help='Режим работы: api - загрузка через API, folder - обработка из папки')
    parser.add_argument('--limit', type=int, default=1,
                        help='Количество изображений для загрузки (только для режима api)')
    parser.add_argument('--operation', choices=['add', 'sub'], default='add',
                        help='Операция: add - сложение, sub - вычитание (только для режима folder)')

    args = parser.parse_args()

    try:
        if args.mode == 'api':
            # Режим API
            print(f"   - Количество изображений: {args.limit}")
            process_from_api(limit=args.limit)

        else:
            # Режим folder
            print(f"   - Операция: {args.operation}")
            process_from_folder(operation=args.operation)

        print("\nПрограмма успешно завершена!")

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")


if __name__ == '__main__':
    main()