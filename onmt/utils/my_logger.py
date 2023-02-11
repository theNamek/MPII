# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function
import logging


def set_logging_process(log_file):
    """
    >>> import logging
    >>> logging.info('msg')
    :param log_file:
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(log_file, 'a')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('Initialized Logger')
    pass




