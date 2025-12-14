import os
import unittest
import numpy as np
from PIL import Image
import tempfile
import asyncio
from unittest.mock import AsyncMock, patch

from async_processing.async_processing import CatImageRGB, CatImageGrayscale, CatImageProcessor


class TestCatImageGrayscale(unittest.TestCase):
    def setUp(self):
        """
        Инициализаия данных для тестов.
        """
        self.rgb_img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.gray_img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        self.gray_img = CatImageGrayscale(self.gray_img_array, "какая-то рандомная ссылка 1", "какая-то порода 1")

    def test_grayscale_to_rgb_succeed(self):
        """
        Проверка корректности преобразования grayscale в RGB
        """
        rgb_from_gray = self.gray_img.rgb_image

        self.assertTrue(
            rgb_from_gray.shape == (64, 64, 3) and
            np.all(rgb_from_gray[:, :, 0] == rgb_from_gray[:, :, 1]) and
            np.all(rgb_from_gray[:, :, 1] == rgb_from_gray[:, :, 2]),
            "RGB изображение должно иметь 3 одинаковых канала"
        )

    def test_image_addition(self):
        """
        Проверка сложения двух grayscale изображений
        """
        other = CatImageGrayscale(self.gray_img_array, "какая-то рандомная ссылка 2", "какая-то порода 2")
        sum_img = self.gray_img + other

        expected = np.clip(self.gray_img_array.astype(np.int16) * 2, 0, 255).astype(np.uint8)

        # ОДИН assert для проверки всего массива
        self.assertTrue(
            np.array_equal(sum_img.image, expected),
            "Сумма изображений должна соответствовать поэлементному сложению"
        )

    def test_custom_edges(self):
        """
        Проверка работы кастомного edge detection на grayscale
        """
        edges = self.gray_img.edges(opencv_realization=False)

        # ОДИН assert проверяет и тип и размер
        self.assertTrue(
            isinstance(edges, CatImageGrayscale) and edges.image.shape == (62, 62),
            "Edge detection должен возвращать CatImageGrayscale с размером (62, 62)"
        )


class TestCatImageRGB(unittest.TestCase):
    def setUp(self):
        """
        Инициализаия данных для тестов.
        """
        self.rgb_img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.gray_img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        self.rgb_img = CatImageRGB(self.rgb_img_array, "какая-то рандомная ссылка 3", "какая-то порода 3")

    def test_rgb_to_grayscale(self):
        """
        Проверка корректности преобразования RGB в grayscale
        """
        gray_from_rgb = self.rgb_img.grayscale_image

        self.assertTrue(
            gray_from_rgb.shape == (64, 64) and gray_from_rgb.dtype == np.uint8,
            "Grayscale преобразование должно давать 2D uint8 массив"
        )

    def test_image_addition(self):
        """
        Проверка сложения двух RGB изображений
        """
        other = CatImageRGB(self.rgb_img_array, "какая-то рандомная ссылка 4", "какая-то порода 4")
        sum_img = self.rgb_img + other

        expected = np.clip(self.rgb_img_array.astype(np.int16) * 2, 0, 255).astype(np.uint8)

        self.assertTrue(
            np.array_equal(sum_img.image, expected),
            "Сумма RGB изображений должна соответствовать поэлементному сложению"
        )

    def test_custom_edges(self):
        """
        Проверка работы кастомного edge detection на RGB и возврата grayscale
        """
        edges = self.rgb_img.edges(opencv_realization=False)

        self.assertTrue(
            isinstance(edges, CatImageGrayscale) and edges.image.shape == (62, 62),
            "Edge detection на RGB должен возвращать CatImageGrayscale с размером (62, 62)"
        )


if __name__ == '__main__':
    unittest.main()