import os
import pytest

def hf0(x):
    return x + 1

def test_hf0_success():
    assert hf0(3) == 4

# def test_hf0_fail():
#     assert hf0(3) == 5

def hf1():
    raise SystemExit(1)

def test_hf1_exception():
    with pytest.raises(SystemExit):
        hf1()


class TestClass(object):
    def test_00(self):
        assert 1+1 == 2

    def not_test(self):
        assert 1==2 #not called in pytest


# def test_texture(tmpdir):
#     print('[my-debug]', type(tmpdir)) #LocalPath??
#     print('[my-debug]', dir(tmpdir))
#     print('[my-debug]', tmpdir)
#     assert False #print above when unittest fail
