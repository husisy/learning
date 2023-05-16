def say():
    print('this is mypackage.say()')

import os
import json
ROOTDIR = os.path.dirname(__file__)
with open(os.path.join(ROOTDIR, '_package.json'), 'r', encoding='utf-8') as fid:
    _package = json.load(fid)

__version__ = _package['version']


class DummyObject:
    '''DummyObject module level doc'''
    def __init__(self) -> None:
        '''DummyObject.__init__ doc'''
        pass

    def print(self):
        '''DummyObject.print doc'''
        pass
