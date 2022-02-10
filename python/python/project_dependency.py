import os
import shutil
import configparser

# just a demo, to use it in project, one need
#    1. modify get_config() as the project required
#    2. add inject dependency part
#    3. write 'project_dependecy.ini.example' and 'project_dependecy.ini'
#    4. (optional) add 'project_dependency.ini' to .gitignore

_ROOT = os.path.dirname(__file__)
assert os.getcwd()==os.path.dirname(__file__), 'project_dependency.py should be at working directory'

_CONFIG_FILE = 'project_dependency.ini'
_CONFIG_EXAMPLE_FILE = 'project_dependency.ini.example'
if not os.path.exists(_CONFIG_FILE):
    print('config file "{}" not exists'.format(_CONFIG_FILE))
    shutil.copyfile(_CONFIG_EXAMPLE_FILE, _CONFIG_FILE)
    print('created from the template "{}"'.format(_CONFIG_EXAMPLE_FILE))
    print('complete the config file "{}" first'.format(_CONFIG_FILE))
    exit()
_CONFIG = configparser.ConfigParser()
_CONFIG.read(_CONFIG_FILE)


def get_config(key:str=None)->str:
    '''
    key(str)
    (ret)(str)
    '''
    xxx_section = _CONFIG['xxx']
    if key is None:
        for key in xxx_section.keys():
            print('{}: {}'.format(key, xxx_section[key]))
    else:
        return xxx_section[key]


# inject dependency (write in project_dependency.py)
'''
import xxx_module
xxx_module.set_default_parameter(xxx=get_config('xxx'))
'''

# xxx_module.py
'''
_DEFAULT_PARAMETER = {
    'xxx': None,
}

def set_default_parameter(xxx=None):
    if xxx is not None:
        _DEFAULT_PARAMETER['xxx'] = xxx
'''
