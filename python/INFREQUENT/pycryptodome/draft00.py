from Cryptodome.Cipher import AES
from Cryptodome.PublicKey import RSA
from Cryptodome.Random import get_random_bytes
from Cryptodome.Cipher import AES, PKCS1_OAEP

hf_bytes_to_binary = lambda x: bin(int(x.hex(), 16))[2:].zfill(len(x)*8) #remove first 0b
hf_bytes_to_binary(b'\x00') #'00000000'
hf_bytes_to_binary(b'\x04') #'00000100'
hf_bytes_to_binary(b'\x10') #'00010000'
hf_bytes_to_binary(b'\x18') #'00011000'
hf_bytes_to_binary(b'\xff') #'11111111'



key = get_random_bytes(16)#bytes
message = 'encrypted this message'.encode('utf-8')
cipher = AES.new(key, AES.MODE_EAX)
nonce = cipher.nonce #IV_str, as same length as key
ciphertext,tag = cipher.encrypt_and_digest(message)
# given key + nonce + ciphertext
decypher_message1 = AES.new(key, AES.MODE_EAX, nonce).decrypt(ciphertext)#bytes
# given key + nonce + ciphertext + tag (for verification)
decypher_message2 = AES.new(key, AES.MODE_EAX, nonce).decrypt_and_verify(ciphertext, tag) #bytes


secret_code = 'Unguessable'
key = RSA.generate(2048)
encrypted_key = key.export_key(passphrase=secret_code, pkcs=8, protection='scryptAndAES128-CBC')
key.publickey().export_key()

# TODO https://www.pycryptodome.org/en/latest/src/examples.html

key = RSA.generate(2048)
private_key_bytes = key.export_key() #not random
public_key_bytes = key.publickey().export_key() #not random

def with_public_key(data, public_key=RSA.import_key(public_key_bytes)):
    #encrypt the session key with the public_key
    session_key = get_random_bytes(16)
    enc_session_key = PKCS1_OAEP.new(public_key).encrypt(session_key) #encrypt the session key with the public_key
    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    ciphertext,tag = cipher_aes.encrypt_and_digest(data)
    return ciphertext, tag, enc_session_key, cipher_aes.nonce

def with_private_key(ciphertext, tag, enc_session_key, nonce, private_key=RSA.import_key(private_key_bytes)):
    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)
    cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
    return cipher_aes.decrypt_and_verify(ciphertext, tag)

data = 'encrypt this message'.encode('utf-8')
data_ = with_private_key(*with_public_key(data))
