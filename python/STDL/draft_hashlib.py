import hmac
import hashlib

md5 = hashlib.md5()
md5.update(b'how to use md5 in python hashlib?') #in python3, default utf-8
md5.hexdigest() #d26a53750bc40b38b65a520292f69306


md5 = hashlib.md5()
md5.update(b'how to use md5 in ')
md5.update(b'python hashlib?')
md5.hexdigest() #d26a53750bc40b38b65a520292f69306


sha1 = hashlib.sha1()
sha1.update(b'how to use md5 in python hashlib?') #in python3, default utf-8
sha1.hexdigest()


# TODO sha256
# TODO sha512
# salt: https://en.wikipedia.org/wiki/Salt_(cryptography)


# Keyed-Hashing for Message Authentication： https://en.wikipedia.org/wiki/HMAC
x1 = hmac.new(key=b'secret', msg=b'hello world', digestmod='MD5')
x1.hexdigest()
