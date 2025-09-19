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

    def _pad_image(self: 'ImageProcessing', image: np.ndarray,
                   pad_h: int, pad_w: int) -> np.ndarray:
        """
        Метод для добавления отступов к изображению

        Args:
            image (np.ndarray): Изображение
            pad_h (int): Отсутпы по вертикали
            pad_w (int): Отступы по горизонтали

        Returns:
            np.ndarray: Изображение с отступами
        """
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    def _convolution(self: 'ImageProcessing', image: np.ndarray,
                     kernel: np.ndarray, padding: str = "same",
                     pad_h: int = 0, pad_w: int = 0) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение (чёрно-белое или цветное).
            kernel (np.ndarray): Ядро свёртки (матрица).
            padding (str): Формат выставления отступов.
            pad_h (int): Отступ по вертикали.
            pad_w (int): Отступ по горизонтали.

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """

        if padding == "same":
            kernel_h, kernel_w = kernel.shape
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            image = self._pad_image(image, pad_h, pad_w)

        kern_h, kern_w = kernel.shape
        img_h, img_w = image.shape[0:2]
        out_h, out_w, out_d = img_h - kern_h + 1, img_w - kern_w + 1, image.ndim
        if image.ndim == 2:
            conv_res = np.zeros((out_h, out_w))
            for heigh in range(out_h):
                for wid in range(out_w):
                    conv_res[heigh, wid] = np.sum(
                        image[heigh:heigh+kern_h, wid:wid+kern_w] * kernel)
        else:
            out_d = image.shape[2]
            conv_res = np.zeros((out_h, out_w, out_d))
            for heigh in range(out_h):
                for wid in range(out_w):
                    for dim in range(out_d):
                        conv_res[heigh, wid, dim] = np.sum(
                            image[heigh:heigh+kern_h, wid:wid+kern_w, dim] * kernel)

        return conv_res

    def _rgb_to_grayscale(self: 'ImageProcessing', image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.

        Raises:
            ValueError: Некорректный формат изображения
        """
        if image.ndim != 3:
            raise ValueError('Incorrect image format. Required format: RGB')

        return (np.dot(image[:, :, :3], [0.299, 0.587, 0.114])).astype(np.uint8)

    def _gamma_correction(self: 'ImageProcessing', image: np.ndarray,
                          gamma: float) -> np.ndarray:
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

        normalized = image / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        corrected = np.uint8(corrected * 255)
        return corrected

    @time_execution
    def edge_detection(self: 'ImageProcessing', image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении с использованием оператора Собеля.
        Возвращает бинарное изображение с яркими белыми границами на черном фоне.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарное изображение с границами (0 - черный, 255 - белый).

        Raises:
            ValueError: Некорректный формат изображения
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Incorrect image format. Required format: RGB (3 channels)')

        # Преобразование в оттенки серого
        grayscale_image = self._rgb_to_grayscale(image)

        # Ядра Собеля для горизонтальных и вертикальных границ
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Свертка с ядрами
        edges_x = self._convolution(grayscale_image, sobel_x, padding="same")
        edges_y = self._convolution(grayscale_image, sobel_y, padding="same")

        # Вычисление магнитуды градиента
        gradient_magnitude = np.hypot(edges_x, edges_y)

        # Автоматическое определение порога (метод Отсу)
        def otsu_threshold(data: np.ndarray) -> float:
            """Реализация алгоритма Отсу для автоматического определения порога."""
            hist, bins = np.histogram(data.flatten(), bins=256, range=[0, 256])
            bin_centers = (bins[:-1] + bins[1:]) / 2

            total = len(data.flatten())
            sum_total = np.sum(bin_centers * hist)

            sum_b = 0
            w_b = 0
            max_variance = 0
            threshold = 0

            for i in range(256):
                w_b += hist[i]
                if w_b == 0:
                    continue

                w_f = total - w_b
                if w_f == 0:
                    break

                sum_b += bin_centers[i] * hist[i]
                m_b = sum_b / w_b
                m_f = (sum_total - sum_b) / w_f

                variance = w_b * w_f * (m_b - m_f) ** 2

                if variance > max_variance:
                    max_variance = variance
                    threshold = bin_centers[i]

            return threshold

        # Применение порога для получения бинарного изображения
        threshold_value = otsu_threshold(gradient_magnitude)

        # Дополнительное усиление: оставляем только самые яркие пиксели
        # Можно регулировать множитель для более/менее агрессивного порога
        enhanced_threshold = threshold_value * 4.0

        # Создание бинарного изображения
        binary_edges = np.zeros_like(gradient_magnitude, dtype=np.uint8)
        binary_edges[gradient_magnitude > enhanced_threshold] = 255

        return binary_edges

    def _gauss_kernel(self: 'ImageProcessing', std: float = 0.5,
                      window_size: int = 3) -> np.ndarray:
        """
        Метод генерации гауссова ядра

        Args:
            std (float, optional): Дисперсия. Defaults to 0.5.
            window_size (int, optional): Размер ядра. Defaults to 3.

        Returns:
            np.ndarray: Матрица с гауссовым ядром
        """
        ax = np.arange(-window_size // 2 + 1., window_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        gauss_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * std ** 2))
        gauss_kernel /= np.sum(gauss_kernel)
        return gauss_kernel

    @time_execution
    def corner_detection(self: 'ImageProcessing', image: np.ndarray,
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
        if image.ndim != 3:
            raise ValueError('Incorrect image format. Required format: RGB')

        grayscale_image = self._rgb_to_grayscale(image)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        edges_x = self._convolution(grayscale_image, sobel_x, padding="same")
        edges_y = self._convolution(grayscale_image, sobel_x.T, padding="same")

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
        angle_feature = det - sensitivity * (trace ** 2)

        result_image = np.stack([grayscale_image] * 3, axis=-1)
        result_image[angle_feature > threshold * angle_feature.max()] = [0, 0, 255]

        return result_image

    @time_execution
    def circle_detection(self: 'ImageProcessing', image: np.ndarray) -> None:
        """
        Метод выделения кругов с использованием алгоритма Хафа

        Args:
            image (np.ndarray): Изображение

        Raises:
            NotImplementedError: Ошибка отсутствия реализации
        """
        raise NotImplementedError()
