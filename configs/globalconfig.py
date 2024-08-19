"""
File: globalconfig.py
Date: 2024/7/31
Author: yruns


"""


class GlobalConfig(object):

    @classmethod
    def log_self(cls, logger=None):
        if logger is None:
            log = print
        else:
            log = logger.info
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                log(f"{attr}: {getattr(cls, attr)}")
