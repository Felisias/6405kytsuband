import os
import unittest
import numpy as np
from PIL import Image
import tempfile
import asyncio
from unittest.mock import AsyncMock, patch
import io

from async_processing.async_processing import CatImageRGB, CatImageGrayscale, CatImageProcessor

class TestCatImageProcessorFileIO(unittest.TestCase):
    def setUp(self):
        """
        Инициализаия данных для тестов.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.processor = CatImageProcessor(output_dir=self.temp_dir)

    def tearDown(self):
        """
        Очистка временных файлов после тестов.
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_image_creates_valid_jpeg_file(self):
        """
        Проверка правильности сохранения в файл.
        """
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        cat_img = CatImageRGB(img_array, "ссылка на кота", "кот")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            file_path = tmp.name

        try:
            asyncio.run(self.processor.save_image(file_path, cat_img))

            # ОДИН assert с корректным закрытием файла
            file_exists = os.path.exists(file_path)
            if file_exists:
                with Image.open(file_path) as img:
                    size_correct = img.size == (32, 32)
            else:
                size_correct = False

            self.assertTrue(
                file_exists and size_correct,
                "Сохраненный файл должен существовать и иметь правильный размер"
            )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_save_images_async_creates_correct_directory_structure(self):
        """
        Проверка сохранения в правильный файл и правильную структуру папок.
        """
        img = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
        cat = CatImageRGB(img, "рандомная ссылка на кота", "рандомная_порода")
        processed = [(0, cat, cat, cat)]

        asyncio.run(self.processor.save_images_async(processed))

        timestamp_dirs = os.listdir(self.temp_dir)
        breed_dir = os.path.join(self.temp_dir, timestamp_dirs[0], "0_рандомная_порода")
        files = set(os.listdir(breed_dir))

        expected_files = {
            "0_рандомная_порода_original.jpg",
            "0_рандомная_порода_custom.jpg",
            "0_рандомная_порода_cv2.jpg"
        }

        # ОДИН assert проверяет всю структуру
        self.assertTrue(
            len(timestamp_dirs) == 1 and
            os.path.isdir(breed_dir) and
            files == expected_files,
            "Должна создаться правильная структура папок и файлов"
        )

    def test_saved_images_are_valid_image_files(self):
        """
        Проверка, что сохраненные файлы являются валидными изображениями.
        """
        img = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
        cat = CatImageRGB(img, "ссылка", "порода")
        processed = [(0, cat, cat, cat)]

        asyncio.run(self.processor.save_images_async(processed))

        timestamp_dirs = os.listdir(self.temp_dir)
        breed_dir = os.path.join(self.temp_dir, timestamp_dirs[0], "0_порода")

        # ОДИН assert с корректным закрытием всех файлов
        all_files_valid = True
        for filename in os.listdir(breed_dir):
            filepath = os.path.join(breed_dir, filename)
            try:
                with Image.open(filepath) as img:
                    if img.format != 'JPEG':
                        all_files_valid = False
                        break
            except Exception:
                all_files_valid = False
                break

        self.assertTrue(
            all_files_valid,
            "Все сохраненные файлы должны быть валидными JPEG изображениями"
        )


class TestCatImageProcessorAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processor = CatImageProcessor(output_dir=self.temp_dir)

    async def asyncTearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('async_processing.async_processing.implementation.cat_image_processor.aiohttp.ClientSession.get')
    async def test_download_data_returns_correct_structure(self, mock_get):
        """
        Тесттирование корректного обращения к API.
        """
        fake_img = Image.new('RGB', (1, 1), color='red')
        img_bytes = io.BytesIO()
        fake_img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()

        mock_response_json = AsyncMock()
        mock_response_json.json.return_value = [{
            "url": "http://example.com/cat123.jpg",
            "breeds": [{"name": "Scottish Fold"}]
        }]

        mock_response_img = AsyncMock()
        mock_response_img.read.return_value = img_bytes

        mock_get.side_effect = [mock_response_json, mock_response_img]
        mock_response_json.__aenter__.return_value = mock_response_json
        mock_response_img.__aenter__.return_value = mock_response_img

        results = await self.processor.download_data(limit=1)

        # ОДИН assert проверяет всю возвращаемую структуру
        self.assertTrue(
            len(results) == 1 and
            results[0][0] == 0 and
            results[0][1].url == "http://example.com/cat123.jpg" and
            results[0][1].breed == "Scottish Fold" and
            results[0][1].image.shape == (1, 1, 3),
            "download_data должен возвращать корректную структуру данных"
        )

    @patch('async_processing.async_processing.implementation.cat_image_processor.aiohttp.ClientSession.get')
    async def test_download_data_makes_correct_api_calls(self, mock_get):
        """
        Проверка что API вызывается с правильными параметрами.
        """
        mock_response_json = AsyncMock()
        mock_response_json.json.return_value = []
        mock_response_json.__aenter__.return_value = mock_response_json
        mock_get.return_value = mock_response_json

        await self.processor.download_data(limit=1)

        # ОДИН assert проверяет все параметры вызова
        self.assertTrue(
            mock_get.called and
            mock_get.call_args[0][0] == "https://api.thecatapi.com/v1/images/search" and
            mock_get.call_args[1]['params']['limit'] == 1 and
            mock_get.call_args[1]['params']['has_breeds'] == 'True',
            "API должно вызываться с правильными параметрами"
        )


if __name__ == '__main__':
    unittest.main()