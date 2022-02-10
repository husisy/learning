from Cryptodome.PublicKey import RSA

def fake_ssh_keygen(public_key_file='id_rsa.pub', private_key_file='id_rsa'):
    # fail
    key = RSA.generate(4096)
    private_key_bytes = key.export_key()
    public_key_bytes = key.publickey().export_key()
    with open(private_key_file, 'w', encoding='utf-8', newline='\n') as fid:
        fid.write(private_key_bytes.decode('utf-8'))
    with open(public_key_file, 'w', encoding='utf-8', newline='\n') as fid:
        fid.write(public_key_bytes.decode('utf-8'))
    print('please copy {} to ~/.ssh/id_rsa'.format(private_key_file))
    print('please copy {} to ~/.ssh/id_rsa.pub'.format(public_key_file))
