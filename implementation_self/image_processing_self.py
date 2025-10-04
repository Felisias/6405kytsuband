"""
Модуль для обработки изображений с использованием numpy
"""
import interfaces

import numpy as np
import time
from functools import wraps

def time_execution(func):
    """Декоратор для измерения времени выполнения методов"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Метод {func.__name__} [self] выполнен за {execution_time:.4f} секунд")
        return result
    return wrapper
class ImageProcessingSelf(interfaces.IImageProcessing):
    """
    Класс обработки изображений с помощью numpy
    """

    def _pad_image(self: 'ImageProcessingSelf', image: np.ndarray,pad_h: int, pad_w: int) -> np.ndarray:
        """
        Метод для добавления отступов к изображению

        Args:
            image (np.ndarray): Изображение
            pad_h (int): Отсутпы по вертикали
            pad_w (int): Отступы по горизонтали

        Returns:
            np.ndarray: Изображение с отступами
        """
        if image.ndim == 2:  # ч/б
            return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        elif image.ndim == 3:  # цветное (RGB)
            return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            raise ValueError("Unsupported image dimensions: expected 2D or 3D array")

    def _convolution(self: 'ImageProcessingSelf',
                     image: np.ndarray,
                     kernel: np.ndarray,
                     padding: str = "same",
                     pad_h: int = 0,
                     pad_w: int = 0) -> np.ndarray:
        """
        Быстрая свёртка через im2col (без циклов).
        """
        if padding == "same":
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            image = self._pad_image(image, pad_h, pad_w)

        H, W = image.shape[:2]
        kh, kw = kernel.shape

        if image.ndim == 2:  # ч/б
            # im2col
            shape = (H - kh + 1, W - kw + 1, kh, kw)
            strides = image.strides * 2
            patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
            patches = patches.reshape(-1, kh * kw)

            print(patches.size)
            conv_res = patches @ kernel.flatten()

            print(f"использую ЧБ")
            return conv_res.reshape(H - kh + 1, W - kw + 1)

        else:  # RGB
            out_channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                shape = (H - kh + 1, W - kw + 1, kh, kw)
                strides = channel.strides * 2
                patches = np.lib.stride_tricks.as_strided(channel, shape=shape, strides=strides)
                patches = patches.reshape(-1, kh * kw)
                conv_res = patches @ kernel.flatten()
                out_channels.append(conv_res.reshape(H - kh + 1, W - kw + 1))

                print(f"использую RGB")
            return np.stack(out_channels, axis=2)

    def _rgb_to_grayscale(self: 'ImageProcessingSelf', image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.

        Raises:
            ValueError: Некорректный формат изображения
        """
        if image.ndim == 2:  # Уже ЧБ
            return image
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB
            return (np.dot(image[:, :, :3], [0.299, 0.587, 0.114])).astype(np.uint8)
        else:
            raise ValueError(f'Unsupported image format: {image.shape}')

    def _gamma_correction(self: 'ImageProcessingSelf', image: np.ndarray,gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.

        Raises:
            ValueError: Некорректные значения gamma
        """
        if gamma <= 0:
            raise ValueError("Incorrect gamma value. Required values > 0")

        normalized = image / 255.0                      # перевод в диапазон от 0 до 1
        corrected = np.power(normalized, 1.0 / gamma)   # степень
        corrected = np.uint8(corrected * 255)           # перевод в диапазон от 0 до 255
        return corrected

    def otsu_threshold(self: 'ImageProcessingSelf', data: np.ndarray) -> float:
        """Реализация алгоритма Отсу для автоматического определения порога."""
        hist, bins = np.histogram(data.flatten(), bins=256, range=[0, 256])  # Изображение в 1D массив
        # hist - массив из 256 элементов, где каждый элемент показывает, сколько пикселей имеет данную яркость

        bin_centers = (bins[:-1] + bins[1:]) / 2

        total = len(data.flatten())  # общее количество пикселей
        sum_total = np.sum(bin_centers * hist)  # сумма всех значений яркости

        sum_b = 0
        w_b = 0
        max_variance = 0
        threshold = 0

        for i in range(256):  # перебираем все возможные пороги от 0 до 255
            w_b += hist[i]  # накопленный вес фона: пиксели ≤ текущего порога
            if w_b == 0:
                continue

            w_f = total - w_b  # вес переднего плана: пиксели > текущего порога
            if w_f == 0:
                break

            sum_b += bin_centers[i] * hist[i]  # накопленная сумма яркостей фона
            m_b = sum_b / w_b  # средняя яркость фона
            m_f = (sum_total - sum_b) / w_f  # средняя яркость переднего плана

            # межклассовая дисперсия максимальна, если классы хорошо разделены и значимы
            variance = w_b * w_f * (m_b - m_f) ** 2

            if variance > max_variance:
                max_variance = variance
                threshold = bin_centers[i]  # запоминаем лучший порог

        return threshold

    def _get_sobel_kernels(self: 'ImageProcessingSelf') -> tuple[np.ndarray, np.ndarray]:
        """Возвращает ядра Собеля для X и Y направлений"""
        sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=np.float32)
        return sobel_x, sobel_y

    def _apply_threshold(self: 'ImageProcessingSelf', gradient_magnitude: np.ndarray, multiplier: float = 4.0) -> np.ndarray:
        """Применяет порог Отсу и создает бинарное изображение"""
        threshold_value = self.otsu_threshold(gradient_magnitude)

        enhanced_threshold = threshold_value * multiplier

        binary_edges = np.zeros_like(gradient_magnitude, dtype=np.uint8)
        binary_edges[gradient_magnitude > enhanced_threshold] = 255

        return binary_edges

    def _compute_gradients(self: 'ImageProcessingSelf', image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вычисляет градиенты по X и Y с помощью ядер Собеля"""
        sobel_x, sobel_y = self._get_sobel_kernels()

        gx = self._convolution(image, sobel_x, padding="same")
        gy = self._convolution(image, sobel_y, padding="same")

        return gx, gy

    def _edge_detection_color(self: 'ImageProcessingSelf', image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        # Используем общий метод для градиентов
        gx, gy = self._compute_gradients(img)

        A = np.sum(gx * gx, axis=2)
        B = np.sum(gx * gy, axis=2)
        C_ = np.sum(gy * gy, axis=2)

        tmp = np.sqrt((A - C_) ** 2 + 4 * (B ** 2))
        lambda_max = 0.5 * (A + C_ + tmp)
        gradient_magnitude = np.sqrt(lambda_max)

        # Метод для порога
        return self._apply_threshold(gradient_magnitude, multiplier=4.5)

    def _edge_detection_grayscale(self: 'ImageProcessingSelf', image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        # Используем общий метод для градиентов
        gx, gy = self._compute_gradients(img)

        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Метод для порога
        return self._apply_threshold(gradient_magnitude, multiplier=4.0)


    @time_execution
    def edge_detection(self: 'ImageProcessingSelf', image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Args:
            image (np.ndarray): Входное изображение (RGB или ЧБ)

        Returns:
            np.ndarray: Бинарное изображение с границами
        """
        if image.ndim == 2:  # ЧБ изображение
            return self._edge_detection_grayscale(image)
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB изображение
            return self._edge_detection_color(image)
        else:
            raise ValueError(f'Unsupported image format: {image.shape}')

    def _gauss_kernel(self: 'ImageProcessingSelf', std: float = 0.5,
                      window_size: int = 3) -> np.ndarray:
        """
        Метод генерации гауссова ядра

        Args:
            std (float, optional): Дисперсия. Defaults to 0.5.
            window_size (int, optional): Размер ядра. Defaults to 3.

        Returns:
            np.ndarray: Матрица с гауссовым ядром
        """
        ax = np.arange(-window_size // 2 + 1., window_size // 2 + 1.)   # Формируем координаты по одной оси для не чётого window_size
        xx, yy = np.meshgrid(ax, ax)    # Двоичная сетка кордов
        gauss_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * std ** 2))
        gauss_kernel /= np.sum(gauss_kernel)    # Нормируем ядро, чтобы сумма элементов =0. Фильтр не должен менять яркость
        return gauss_kernel

    @time_execution
    def corner_detection(self: 'ImageProcessingSelf', image: np.ndarray,
                         sensitivity: float = 0.04,
                         threshold: float = 0.01) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении с помощью детектора Харриса.

        Args:
            image (np.ndarray): Входное изображение (RGB).
            sensitivity (float): Параметр чувствительности детектора.
            threshold (float): Порог для отсечения слабых углов.

        Returns:
            np.ndarray: Изображение с выделенными углами.

        Raises:
            ValueError: Некорректный формат изображения
        """
        if image.ndim == 2:  # ЧБ изображение
            grayscale_image = image
        elif image.ndim == 3:  # RGB изображение
            grayscale_image = self._rgb_to_grayscale(image)
        else:
            raise ValueError(f'Unsupported image format: {image.shape}')

        sobel_x, sobel_y = self._get_sobel_kernels()

        edges_x = self._convolution(grayscale_image, sobel_x, padding="same")
        edges_y = self._convolution(grayscale_image, sobel_y, padding="same")

        edges_xx = edges_x * edges_x
        edges_yy = edges_y * edges_y
        edges_xy = edges_x * edges_y

        window_size = 3
        gauss_kernel = self._gauss_kernel(std=1.0, window_size=window_size)

        smoothed_xx = self._convolution(edges_xx, gauss_kernel, padding="same")
        smoothed_yy = self._convolution(edges_yy, gauss_kernel, padding="same")
        smoothed_xy = self._convolution(edges_xy, gauss_kernel, padding="same")

        det = smoothed_xx * smoothed_yy - smoothed_xy * smoothed_xy
        trace = smoothed_xx + smoothed_yy
        angle_feature = det - sensitivity * (trace ** 2)    # Харис

        result_image = np.stack([grayscale_image] * 3, axis=-1)     # Взращаемся к RGB
        result_image[angle_feature > threshold * angle_feature.max()] = [0, 0, 255]     # Отмечаем точки

        return result_image

    @time_execution
    def circle_detection(self: 'ImageProcessingSelf', image: np.ndarray) -> None:
        """
        Метод выделения кругов с использованием алгоритма Хафа

        Args:
            image (np.ndarray): Изображение

        Raises:
            NotImplementedError: Ошибка отсутствия реализации
        """
        raise NotImplementedError()
