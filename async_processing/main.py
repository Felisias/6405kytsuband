"""
Модуль для консольного взаимодеуствия с методами обработки изображения
"""
import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from async_processing_lib import CatImageProcessor, setup_logger

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default=".")
    parser.add_argument("--log-file", type=str, default="app.log")
    args = parser.parse_args()

    setup_logger(log_dir=args.log_dir, log_file=args.log_file)
    processor = CatImageProcessor(logging_path=args.log_file, logging_dir=args.log_dir)

    try:
        await processor.run_async(limit=args.limit)
        logging.info("Application finished successfully.")
    except Exception as e:
        logging.error("Application failed with error: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())