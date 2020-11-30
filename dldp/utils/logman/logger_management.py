#! /home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: lesion based prediction
==============================
PEP: 8
Author: Weizhe Li <email at example.com>
Sponsor: * Weijie Chen <email at example.com>
BDFL-Delegate:
Discussions-To: *[...]
Status: Draft
Type: [Standards Track | Informational | Process]
Content-Type: text/x-rst
Requires: *[NNN]
Created: 10-31-2019
Python-Version: 3.5
Post-History:
Resolution:
Description:
------------
    This module is used to set up logging files

    Input: The stdout and stderr, other place has the entries for INFO, DEBUG, ERROR, WARN, etc.
    Output: logging files under /tmp
    This module should be run before the module generating FROC curve.
"""
import os
import logging
import logging.config
import sys
import yaml


class debugFilter(logging.Filter):
    """
    a filter class
    """
    def filter(self, rec):
        return rec.levelno == logging.DEBUG

class warnFilter(logging.Filter):
    """
    a filter class
    """
    def filter(self, rec):
        return rec.levelno == logging.WARN


def setup_logging(template_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Logging Setup

    :param template_path: Logging configuration path
    :type template_path: str
    :param default_level: Default logging level
    :type default_level: str
    :param env_key: Logging config path set in environment variable
    :type str
    :return no return

    """
    path = template_path
    print(path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print('Error in loading logging Configuration.', e)
                logging.basicConfig(level=default_level, stream=sys.stdout)
    else:
        logging.basicConfig(level=default_level, stream=sys.stdout)
        print('No logging configure file exists.')


class StreamToLogger(object):
    """
    redirect stdout and stderr to logging files
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def log_with_template(template_path, module_name):
    """
    The function is for setting up log files with stdout and stderr redirected to INFO section of the logging

    :param template_path:
    :type template_path: str
    :return: no return
    """
    # Set up the logging files
    setup_logging(template_path=template_path)
    logger = logging.getLogger(module_name)
    # Once the log files are created, no new log file will be created anymore.
    logger.setLevel(logging.DEBUG)
    #logger_child = logger.getChild("test")
    stdout_logger = logging.getLogger('%s_STDOUT' % module_name)
    sl_out = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl_out

    stderr_logger = logging.getLogger('%s_STDERR' % module_name)
    sl_error = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl_error

    return logger
