import logging

import coloredlogs

# 初始化logger
_logger = logging.getLogger("mentabotix")
coloredlogs.install(logger=_logger, level=logging.DEBUG)


def set_log_level(level: int | str):
    """
    设置日志级别
    :param level: 日志级别
    :return:
    """
    _logger.setLevel(level)


set_log_level(logging.INFO)

if __name__ == "__main__":

    _logger.debug("This is a debug log.")
    _logger.info("This is a info log.")
    _logger.warning("This is a warning log.")
    _logger.error("This is a error log.")
    _logger.critical("This is a critical log.")
