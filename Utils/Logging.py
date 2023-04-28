import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    fmt="%(asctime)s %(levelname)s %(message)s"#一些预置的方式记录文件
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt,datefmt)

    logger = logging.getLogger(name)#根据名字返回同一个log对象
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")#日志写到文件中
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #sh = logging.StreamHandler()
    #sh.setFormatter(formatter)
    #logger.addHandler(sh)

    return logger
