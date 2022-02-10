def test_class_bool():
    # see https://docs.python.org/3/reference/datamodel.html#object.__bool__
    class MyClass00(object):
        def __init__(self, value):
            self.value = value
        def __bool__(self):
            return not self.value

    assert bool(MyClass00(True))==False
    assert bool(MyClass00(False))==True
    if not MyClass00(True):
        assert True
    if MyClass00(False):
        assert True
