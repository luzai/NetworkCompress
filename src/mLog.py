import logging,sys

# logging.basicConfig(filename='output/net2net.log', level=logging.DEBUG)

logger = logging.getLogger('net2net')

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s -  %(levelname)s -------- \n\t%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
