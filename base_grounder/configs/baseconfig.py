"""
File: globalconfig.py
Date: 2024/7/31
Author: yruns


"""


class BaseConfig(object):

    @classmethod
    def log_self(cls, logger):
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                logger.info(f"{attr}: {getattr(cls, attr)}")
