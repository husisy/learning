def write_string(path, str0):
    with open(path, 'w') as fid:
        fid.write(str0)

def write_binary(path, str0):
    with open(path, 'wb') as fid:
        fid.write(str0.encode('utf-8'))

def read_string(path):
    with open(path, 'r', encoding='utf-8') as fid:
        ret = fid.read()
    return ret

def read_binary(path):
    with open(path, 'rb') as fid:
        ret = fid.read().decode('utf-8')
    return ret

def test_string_binary_io(tmpdir, key='2333'):
    path = tmpdir.join('tbd00.txt')

    write_string(path, key)
    assert read_string(path)==key
    assert read_binary(path)==key

    write_binary(path, key)
    assert read_string(path)==key
    assert read_binary(path)==key
