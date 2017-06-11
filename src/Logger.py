import logging,sys,os


format='%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)

if os.path.exists('../output') == False:
    os.mkdir('../output')

infoLogName = '../output/net2net.log'
infoLogger = logging.getLogger("infoLog")
infoLogger.setLevel(logging.INFO)
infoHandler = logging.FileHandler(infoLogName, 'w')
infoHandler.setLevel(logging.INFO)
infoHandler.setFormatter(formatter)
infoLogger.addHandler(infoHandler)
logger = infoLogger

errorLogName = '../output/error.log'
errorLogger = logging.getLogger("errorLog")
errorLogger.setLevel(logging.ERROR)
errorHandler = logging.FileHandler(errorLogName, 'w')
errorHandler.setLevel(logging.ERROR)
errorHandler.setFormatter(formatter)
errorLogger.addHandler(errorHandler)

