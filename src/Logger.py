import logging,sys,os
# logging.basicConfig(filename='output/net2net.log', level=logging.DEBUG)
logger = logging.getLogger('net2net')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s ==> %(message)s') #%(asctime)s
ch.setFormatter(formatter)
logger.addHandler(ch)