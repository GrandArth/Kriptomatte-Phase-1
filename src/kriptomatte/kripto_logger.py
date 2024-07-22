import logging
import os.path
from logging import StreamHandler
from logging.handlers import RotatingFileHandler


def Setup_Logger(exr_file_path):
    formatter = logging.Formatter('%(name)-12s: %(levelname) -8s %(message)s')
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    try:
        logging_file_path = os.path.join(os.path.dirname(exr_file_path), "Testing.log")
        file_handler = RotatingFileHandler(f"{logging_file_path}", "a", maxBytes=1024 * 1024 * 5, backupCount=1)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    except PermissionError:
        logging.critical('Permission Denied when writing log to file. Chang the current working directory may help.')
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

