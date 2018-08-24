import logging


def create_logger(logfile, logname):
    FORMAT = '%(asctime)-15s %(message)s'
    formatter = logging.Formatter(FORMAT)
    logger = logging.getLogger(logname)
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
