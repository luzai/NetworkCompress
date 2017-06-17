import logging
import os


# log level: CRITICAL > ERROR > WARNING > INFO > DEBUG
class MyFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


# log dir: ../output/net2net.log & ../output/error.log
if os.path.exists('../output') == False:
    os.mkdir('../output')

# we only need one log, and three handlers
# log name is GAlogger
logger = logging.getLogger('GAlogger')
logger.setLevel(logging.DEBUG)
# set logger format
format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)

# infoHandler only show info about GA infos
infoHandler = logging.FileHandler('../output/net2net.log')
infoHandler.setLevel(logging.INFO)
infoHandler.addFilter(MyFilter(logging.INFO))  # just show INFO logs
infoHandler.setFormatter(formatter)
logger.addHandler(infoHandler)

# streamHandler show all logs into stdout screen, thus maybe easy for debug
# streamHandler only added when debug = True
dbg = False  # TODO: read dbg from gl_config
if dbg == True:
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

# errorHandler only record WARNING and ERROR and CRITICAL
errorHandler = logging.FileHandler('../output/error.log')
errorHandler.setLevel(logging.WARNING)
errorHandler.setFormatter(formatter)
logger.addHandler(errorHandler)
