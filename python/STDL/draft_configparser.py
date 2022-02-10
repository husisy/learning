import os
import configparser

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

config = configparser.ConfigParser()
config['default'] = {
    'ServerAliveInterval': '45',
    'Compression': 'yes',
    'CompressionLevel': '9',
}
config['default']['ForwardX11'] = 'yes'
config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'
config['topsecret.server.com'] = {}
tmp0 = config['topsecret.server.com'] #type(tmp0) is NOT dict
tmp0['Port'] = '50022'
tmp0['ForwardX11'] = 'no'
with open(hf_file('tbd00.ini'), 'w', encoding='utf-8') as fid:
    config.write(fid)

z0 = configparser.ConfigParser()
z0.read(hf_file('tbd00.ini'))
