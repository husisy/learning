import os


def hf_text_data(filename):
    x0 = os.path.join('data', filename)
    assert os.path.exists(x0), f"File {x0} does not exist."
    with open(x0, 'r', encoding='utf-8') as fid:
        text = fid.read()
    return text
