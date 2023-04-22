import os
import base64
import random
import string

import cryptography.fernet
import cryptography.hazmat.primitives.kdf.pbkdf2

def random_string(len_str, seed=None):
    rng = random.Random(seed)
    ret = ''.join(rng.choice(string.printable) for _ in range(len_str))
    return ret


# symmetric encryption recipe
key = cryptography.fernet.Fernet.generate_key() #URL-safe base64-encoded
assert len(base64.urlsafe_b64decode(key))==32 #32 bytes
cipher = cryptography.fernet.Fernet(key)
plaintext = random_string(16).encode('utf-8')
ciphertext = cipher.encrypt(plaintext) #URL-safe base64-encoded
assert plaintext==cipher.decrypt(ciphertext)


key_list = [cryptography.fernet.Fernet.generate_key() for _ in range(2)]
cipher_list = [cryptography.fernet.Fernet(x) for x in key_list]
cipher_multi = cryptography.fernet.MultiFernet(cipher_list)
plaintext = random_string(16).encode('utf-8')
ciphertext = cipher_multi.encrypt(plaintext) #encrypted with the first cipher
assert plaintext==cipher_multi.decrypt(ciphertext) #decrypted one by one


password = random_string(16).encode('utf-8')
salt = os.urandom(16)
tmp0 = cryptography.hazmat.primitives.hashes.SHA256()
kdf = cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC(algorithm=tmp0, length=32, salt=salt, iterations=480000)
# 480000 iterations is recommended by Django
key = base64.urlsafe_b64encode(kdf.derive(password))
cipher = cryptography.fernet.Fernet(key)
