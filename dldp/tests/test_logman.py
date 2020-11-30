import logging
from logger_management import setup_logging
from logger_management import StreamToLogger
import numpy as np
import pandas as pd
import sys
if __name__ == '__main__':

    # Set up the logging files
    # Should be the first statement in the module to avoid circular dependency issues.

    setup_logging(template_path='/home/wli/PycharmProjects/Camelyon2016DeepLearning_GoodScience/logging_config.yaml')
    module_name = sys.modules['__main__'].__file__
    logger = logging.getLogger(module_name)
    # Once the log files are created, no new log file will be created anymore.
    logger.debug("another logging file :D")
    logger.setLevel(logging.DEBUG)
    #logger_child = logger.getChild("test")
    stdout_logger = logging.getLogger('STDOUT')
    sl_out = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl_out

    stderr_logger = logging.getLogger('STDERR')
    sl_error = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl_error
    logger.critical("this is critical!")
    a = np.arange(0, 20, 2)
    logger.info('set value to a: %s' % a)
    b = a*2
    try:
        print(c)

    except Exception as e:

        logger.warn("here is the potential problem", exc_info = True)

    print("hello, catch it on screen")
    print(f)

