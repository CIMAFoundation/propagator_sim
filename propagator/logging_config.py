import logging
import sys

#logging configuration
class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.INFO)
h1.addFilter(InfoFilter())
h2 = logging.StreamHandler()
h2.setLevel(logging.WARNING)
logger.addHandler(h1)
logger.addHandler(h2)
