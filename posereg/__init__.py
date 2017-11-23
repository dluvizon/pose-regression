import sys
if sys.version_info[0] < 3:
    sys.stderr.write('This package was not tested on Python 2.\n')
    sys.stderr.write('It is better to use Python 3!\n')

from .network import build
from .measures import pckh
from .pose import pa16j
