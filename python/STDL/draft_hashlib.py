import os
import io
import base64
import string
import hmac
import hashlib
import secrets

# python -c 'import secrets; print(secrets.token_hex())'
secrets.token_hex() #str f5dc38009fe56be1ff2da109736483df61e92731c46e0a9a7ad9a83c665f607c
secrets.token_hex(16) #2c125e7d9a787edcd1968890ba4fa6e6
# 16bytes=32 hex-string
secrets.token_bytes()
secrets.token_urlsafe(16) #atiRC69wzGUpYoVpDbmAfQ

tmp0 = string.ascii_letters + string.digits
password = ''.join((secrets.choice(tmp0) for _ in range(16)))
salt = os.urandom(16)
# use cryptography instead, not use hashlib.scrypt/hashlib.pbkdf2_hmac
# cryptography: https://cryptography.io/en/latest/hazmat/primitives/key-derivation-functions/#cryptography.hazmat.primitives.kdf.scrypt.Scrypt
# _RECOMMENDED_ITERATIONS = 500_000 #https://docs.python.org/3/library/hashlib.html#key-derivation
# x0 = hashlib.scrypt(password.encode('utf-8'), salt, _RECOMMENDED_ITERATIONS)
# hashlib.pbkdf2_hmac is deprecated
# x0.hex()

hashlib.algorithms_available
# sha512 sm3 blake2b sha512_256 sha3_384 sha3_512 sha3_224 shake_128 md5-sha1
# sha256 ripemd160 sha512_224 sha1 shake_256 blake2s md5 sha3_256 sha384 sha224
hashlib.sha1
hashlib.sha224
hashlib.sha256
hashlib.sha384
hashlib.sha512
hashlib.blake2b
hashlib.blake2s
hashlib.md5
hashlib.sha3_224
hashlib.sha3_256
hashlib.sha3_384
hashlib.sha3_512
hashlib.shake_128
hashlib.shake_256


x0 = hashlib.md5()
x0.update(b'aaa bbb?') #utf-8 encoded bytes
x0.hexdigest() #str 'f5c7d06d15d04b671caf12806408d414'
x0.digest() #bytes b'\xf5\xc7\xd0m\x15\xd0Kg\x1c\xaf\x12\x80d\x08\xd4\x14'
assert x0.digest_size==16 #size in bytes
assert x0.block_size==64 #internal block of the hash algorithm in bytes
x0.name


x0 = hashlib.md5()
x0.update(b'aaa ')
x0.update(b'bbb?')
x0.hexdigest() #str 'f5c7d06d15d04b671caf12806408d414'


# python-3.11
# with open(hashlib.__file__, 'rb') as fid:
#     x0 = hashlib.file_digest(fid, 'md5')
# x0.hexdigest() #str 'f5c7d06d15d04b671caf12806408d414'


x0 = hashlib.new('md5')
x0.update(b'aaa bbb?') #utf-8 encoded bytes
x0.hexdigest() #str 'f5c7d06d15d04b671caf12806408d414'


x0 = hashlib.sha1()
x0.update(b'aaa bbb?')
x0.hexdigest() #str 468de5dff617db0d90322cd3ee6dc1985de8af7e


# TODO sha256
# TODO sha512
# salt: https://en.wikipedia.org/wiki/Salt_(cryptography)


# Keyed-Hashing for Message Authentication： https://en.wikipedia.org/wiki/HMAC
x1 = hmac.new(key=b'secret', msg=b'hello world', digestmod='MD5')
x1.hexdigest()
