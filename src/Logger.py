import logging
import os
import sys

formatter = logging.Formatter('%(levelname)s ==> %(message)s\t[%(filename)s line:%(lineno)d %(asctime)s]')

if not os.path.exists('../output'):
    os.mkdir('../output')

infoLogName = '../output/net2net.log'
infoLogger = logging.getLogger("infoLog")
infoLogger.setLevel(logging.INFO)
infoHandler = logging.FileHandler(infoLogName, 'w')
infoHandler.setLevel(logging.INFO)
infoHandler.setFormatter(formatter)
infoLogger.addHandler(infoHandler)

stdoutHandler = logging.StreamHandler(sys.stdout)
stdoutHandler.setLevel(logging.INFO)
stdoutHandler.setFormatter(formatter)
infoLogger.addHandler(stdoutHandler)

logger = infoLogger

errorLogName = '../output/error.log'
errorLogger = logging.getLogger("errorLog")
errorLogger.setLevel(logging.ERROR)
errorHandler = logging.FileHandler(errorLogName, 'w')
errorHandler.setLevel(logging.ERROR)
errorHandler.setFormatter(formatter)
errorLogger.addHandler(errorHandler)
