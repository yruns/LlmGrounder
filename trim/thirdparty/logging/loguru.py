from trim.utils.comm import is_main_process
from loguru import logger
import sys


def log_in_main_process_only(record):
    return is_main_process()

def default_logger():
    logger.remove()
    logger.add(sys.stdout, filter=log_in_main_process_only, format='[{time:YYYY-MM-DD HH:mm:ss} '
                                  '{file} line {line}] {message}')
    return logger

def custom_logger(**kwargs):
    logger.remove()
    logger.add(**kwargs)
    return logger

default_logger = default_logger()