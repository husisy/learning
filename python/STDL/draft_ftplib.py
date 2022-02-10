import os
import ftplib
import configparser

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

assert os.path.exists('private_info.ini'), 'create a private_info.ini from private_info.ini.example'
z0 = configparser.ConfigParser()
z0.read('private_info.ini')
server_ip = z0['ftplib']['ip']
server_port = int(z0['ftplib']['port'])
username = z0['ftplib']['username']
password = z0['ftplib']['password']

def ftp_upload(ftp, localpath, remotepath):
    with open(localpath, 'rb') as fid:
        ftp.storbinary(f'STOR {remotepath}', fid)

def ftp_download(ftp, remotepath, localpath):
    with open(localpath, 'wb') as fid:
        ftp.retrbinary(f'RETR {remotepath}', fid.write)

key = 2333333333
with open(hf_file('tbd233.txt'), 'w', encoding='utf-8') as fid:
    fid.write(str(key))

with ftplib.FTP() as ftp:
    ftp.connect(host=server_ip, port=server_port)
    ftp.login(user=username, passwd=password)
    ftp.set_pasv(False) #required
    if 'tbd00' not in ftp.nlst():
        ftp.mkd('tbd00')
    ftp_upload(ftp, hf_file('tbd233.txt'), 'tbd00/tbd233.txt')
    ftp_download(ftp, 'tbd00/tbd233.txt', hf_file('tbd234.txt'))
    with open(hf_file('tbd234.txt')) as fid:
        key_again = fid.read()
    print(key_again)

# ftp.quit()
