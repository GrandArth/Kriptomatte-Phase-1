import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler


def Setup_Logger():
    formatter = logging.Formatter('%(name)-12s: %(levelname) -8s %(message)s')
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler("Testing.log", "a", maxBytes=1024 * 1024 * 5, backupCount=1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
    return root_logger
