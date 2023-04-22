# cryptography

1. link
   * [github](https://github.com/pyca/cryptography)
   * [documentation](https://cryptography.io/en/latest/)
   * cryptopals crypto challenges [link](https://cryptopals.com/)
   * crypto 101 [link](https://www.crypto101.io/)
   * NaCl: networking and cryptography library (pronounced "salt") [link](https://nacl.cr.yp.to/) [pynacal/documentation](https://pynacl.readthedocs.io/en/latest/)
   * URL encoding, URL safe [wiki](https://en.wikipedia.org/wiki/URL_encoding)
   * binary-to-text encoding scheme [wiki](https://en.wikipedia.org/wiki/Binary-to-text_encoding): `base64`
2. install
   * `mamba install cryptography`
   * `pip install cryptography`
3. `cryptography.hazmat`: admonition
4. Fernet
   * AES (advanced encryption standard) in CBC (cipher block chain) mode with a 128-bit key for encryption, using PKCS7 padding
   * HMAC (hash-based message authentication code) using SHA256 (secure hash algorithm) for authentication
   * initialization vectors (salt) are generated using `os.urandom()`
   * unsuitable for very large files
5. X.509
