import platform

import oqs
import oqs.rand

oqs.oqs_version() #0.8.0-dev
oqs.oqs_python_version() #0.7.2


kems = oqs.get_enabled_KEM_mechanisms()
# ['BIKE-L1', 'BIKE-L3', 'BIKE-L5', 'Classic-McEliece-348864',
#  'Classic-McEliece-348864f', 'Classic-McEliece-460896',
#  'Classic-McEliece-460896f', 'Classic-McEliece-6688128',
#  'Classic-McEliece-6688128f', 'Classic-McEliece-6960119',
#  'Classic-McEliece-6960119f', 'Classic-McEliece-8192128',
#  'Classic-McEliece-8192128f', 'HQC-128', 'HQC-192', 'HQC-256', 'Kyber512',
#  'Kyber768', 'Kyber1024', 'Kyber512-90s', 'Kyber768-90s', 'Kyber1024-90s',
#  'sntrup761', 'FrodoKEM-640-AES', 'FrodoKEM-640-SHAKE', 'FrodoKEM-976-AES',
#  'FrodoKEM-976-SHAKE', 'FrodoKEM-1344-AES', 'FrodoKEM-1344-SHAKE']

sigs = oqs.get_enabled_sig_mechanisms()
# ['Dilithium2', 'Dilithium3', 'Dilithium5', 'Dilithium2-AES', 'Dilithium3-AES', 'Dilithium5-AES',
#  'Falcon-512', 'Falcon-1024',
#  'SPHINCS+-Haraka-128f-robust', 'SPHINCS+-Haraka-128f-simple', 'SPHINCS+-Haraka-128s-robust',
#  'SPHINCS+-Haraka-128s-simple', 'SPHINCS+-Haraka-192f-robust', 'SPHINCS+-Haraka-192f-simple',
#  'SPHINCS+-Haraka-192s-robust', 'SPHINCS+-Haraka-192s-simple', 'SPHINCS+-Haraka-256f-robust',
#  'SPHINCS+-Haraka-256f-simple', 'SPHINCS+-Haraka-256s-robust', 'SPHINCS+-Haraka-256s-simple',
#  'SPHINCS+-SHA256-128f-robust', 'SPHINCS+-SHA256-128f-simple', 'SPHINCS+-SHA256-128s-robust',
#  'SPHINCS+-SHA256-128s-simple', 'SPHINCS+-SHA256-192f-robust', 'SPHINCS+-SHA256-192f-simple',
#  'SPHINCS+-SHA256-192s-robust', 'SPHINCS+-SHA256-192s-simple', 'SPHINCS+-SHA256-256f-robust',
#  'SPHINCS+-SHA256-256f-simple', 'SPHINCS+-SHA256-256s-robust', 'SPHINCS+-SHA256-256s-simple',
#  'SPHINCS+-SHAKE256-128f-robust', 'SPHINCS+-SHAKE256-128f-simple', 'SPHINCS+-SHAKE256-128s-robust',
#  'SPHINCS+-SHAKE256-128s-simple', 'SPHINCS+-SHAKE256-192f-robust', 'SPHINCS+-SHAKE256-192f-simple',
#  'SPHINCS+-SHAKE256-192s-robust', 'SPHINCS+-SHAKE256-192s-simple', 'SPHINCS+-SHAKE256-256f-robust',
#  'SPHINCS+-SHAKE256-256f-simple', 'SPHINCS+-SHAKE256-256s-robust', 'SPHINCS+-SHAKE256-256s-simple']

## key encapsulation example
# create client and server with sample KEM mechanisms
kemalg = "Kyber512"
with oqs.KeyEncapsulation(kemalg) as client:
    with oqs.KeyEncapsulation(kemalg) as server:
        print("\nKey encapsulation details:")
        print(client.details)
        # {
        #     'name': 'Kyber512',
        #     'version': 'https://github.com/pq-crystals/kyber/commit/74cad307858b61e434490c75f812cb9b9ef7279b',
        #     'claimed_nist_level': 1,
        #     'is_ind_cca': True,
        #     'length_public_key': 800,
        #     'length_secret_key': 1632,
        #     'length_ciphertext': 768,
        #     'length_shared_secret': 32,
        # }

        # client generates its keypair
        public_key = client.generate_keypair()
        # optionally, the secret key can be obtained by calling export_secret_key()
        # and the client can later be re-instantiated with the key pair:
        # secret_key = client.export_secret_key()
        # store key pair, wait... (session resumption):
        # client = oqs.KeyEncapsulation(kemalg, secret_key)

        # the server encapsulates its secret using the client's public key
        ciphertext, shared_secret_server = server.encap_secret(public_key)

        # the client decapsulates the server's ciphertext to obtain the shared secret
        shared_secret_client = client.decap_secret(ciphertext)
        assert shared_secret_client == shared_secret_server #bytes


## various randomness RNGs example
# set the entropy seed to some random values
entropy_seed = [0] * 48
entropy_seed[0] = 100
entropy_seed[20] = 200
entropy_seed[47] = 150
oqs.rand.randombytes_nist_kat_init_256bit(bytes(entropy_seed))
oqs.rand.randombytes_switch_algorithm("NIST-KAT")
print('{:17s}'.format("NIST-KAT:"), ' '.join('{:02X}'.format(x) for x in oqs.rand.randombytes(32)))

# we do not yet support OpenSSL under Windows
if platform.system() != "Windows":
    oqs.rand.randombytes_switch_algorithm("OpenSSL")
    print('{:17s}'.format("OpenSSL:"), ' '.join('{:02X}'.format(x) for x in oqs.rand.randombytes(32)))

oqs.rand.randombytes_switch_algorithm("system")
print('{:17s}'.format("System (default):"), ' '.join('{:02X}'.format(x) for x in oqs.rand.randombytes(32)))


## signature example
message = "This is the message to sign".encode()
# create signer and verifier with sample signature mechanisms
sigalg = "Dilithium2"
with oqs.Signature(sigalg) as signer:
    with oqs.Signature(sigalg) as verifier:
        print(signer.details)
        # {
        #     'name': 'Dilithium2',
        #     'version': 'https://github.com/pq-crystals/dilithium/commit/d9c885d3f2e11c05529eeeb7d70d808c972b8409',
        #     'claimed_nist_level': 2,
        #     'is_euf_cma': True,
        #     'length_public_key': 1312,
        #     'length_secret_key': 2528,
        #     'length_signature': 2420,
        # }

        # signer generates its keypair
        signer_public_key = signer.generate_keypair()
        # optionally, the secret key can be obtained by calling export_secret_key()
        # and the signer can later be re-instantiated with the key pair:
        # secret_key = signer.export_secret_key()
        # store key pair, wait... (session resumption):
        # signer = oqs.Signature(sigalg, secret_key)

        # signer signs the message
        signature = signer.sign(message)

        # verifier verifies the signature
        is_valid = verifier.verify(message, signature, signer_public_key)

        print("\nValid signature?", is_valid)
