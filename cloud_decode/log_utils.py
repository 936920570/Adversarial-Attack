# coding=utf-8
import logging
import functools
import traceback


def get_logger(log_name: str = "default_logger", log_level=logging.INFO):
    # get the logger
    __format_string__ = '%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s'
    __datefmt_string__ = '%Y-%m-%d %H:%M:%S'
    __formatter__ = logging.Formatter(__format_string__, datefmt=__datefmt_string__)
    logger = logging.getLogger(log_name)
    logger.propagate = False
    logger.setLevel(log_level)
    if not logger.hasHandlers():
        __stream_handler__ = logging.StreamHandler()
        __stream_handler__.setFormatter(__formatter__)
        logger.addHandler(__stream_handler__)
    return logger


def exception_printer(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyboardInterrupt as _err:
            raise _err
        except Exception as _err:
            logger = get_logger("exception_logger")
            logger.error("Script Error.")
            traceback.print_exc()

    return wrapper
