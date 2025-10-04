"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки OpenCV.
Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: базовый путь для сохранения результатов (по умолчанию: <имя_входного_файла>_result)

Пример:
    python main.py edges input.jpg
    python main.py corners input.jpg -o corners_result

Автор: [Ваше имя]
"""

import argparse
import os

import cv2

from implementation import ImageProcessing
from implementation_self import ImageProcessingSelf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV и собственная реализация).",
    )
    parser.add_argument(
        "method",
        choices=[
            "edges",
            "corners",
            "circles",
            "gray"
        ],
        help="Метод обработки: edges, corners, circles",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Базовый путь для сохранения результатов (по умолчанию: <input>_result)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    #image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    processor = ImageProcessing()
    processor_self = ImageProcessingSelf()

    # Определение базового пути для сохранения
    if args.output:
        base_output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        base_output_path = f"{base}_result"

    # Обработка
    if args.method == "edges":
        result_cv2 = processor.edge_detection(image)
        result_self = processor_self.edge_detection(image)
    elif args.method == "corners":
        result_cv2 = processor.corner_detection(image)
        result_self = processor_self.corner_detection(image)
    elif args.method == "gray":
        result_self = processor_self._rgb_to_grayscale(image)
    elif args.method == "circles":
        result_cv2 = processor.circle_detection(image)
        result_self = processor_self.circle_detection(image)
    else:
        print("Ошибка: неизвестный метод")
        return

    # Сохранение результатов
    output_cv2 = f"{base_output_path}_cv2.png"
    output_self = f"{base_output_path}_self.png"

    cv2.imwrite(output_cv2, result_cv2)
    cv2.imwrite(output_self, result_self)

    print(f"Результат OpenCV сохранён в {output_cv2}")
    print(f"Результат собственной реализации сохранён в {output_self}")


if __name__ == "__main__":
    main()