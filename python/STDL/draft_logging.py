'''
print: ordinary usage of a command line script or program
logging.info(): events for normal operation of a program (status monitoring, investigation)
logging.debug(): similar to .info(), but for very detailed output for diagonstic purposes
warnings.warn(): a warning regarding a particular runtime event
logging.warning()
logging.error()
logging.exception()
logging.critical()
'''

'''level
DEBUG
INFO
WARNING(default)
ERROR
CRITICAL
'''

import os
import logging

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

tmp0 = {
    'filename': hf_file('tbd00.txt'),
    'format': '%(asctime)s:%(levelname)s:%(message)s',
    # 'format': '[%(levelname)s]%(asctime)s:%(message)s',
    'level': logging.DEBUG,
    'filemode': 'a',
    'datefmt': '%Y%m%d %H:%M:%S',
}
logging.basicConfig(**tmp0)

logging.debug('debug233')
logging.info('info234')
logging.warning('warning235')
logging.getLogger().level #DEBUG=10

# TODO getLoggger
# TODO logger.handles[0].close()

def hf1():
    raise RuntimeError('hf1():233')

def hf2():
    hf1()

try:
    hf2()
except Exception as x1:
    logging.exception(x1)
