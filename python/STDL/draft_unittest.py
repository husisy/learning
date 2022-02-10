import unittest
from unittest.mock import MagicMock, Mock

class TestStringMethods(unittest.TestCase):
    def test_whatya(self):
        self.assertEqual('foo'.upper(), 'FOO')
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
        self.assertEqual('hello world'.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            'hello world'.split(2)
    def not_run(self):
        self.assertTrue('233'=='332')


class ZC0(object):
    def hf1(self, a, b):
        # assuming not finished
        pass

class ZC0Unittest(unittest.TestCase):

    def test_hf1(self):
        x0 = ZC0()
        x0.hf1 = Mock(return_value=13)
        self.assertEqual(x0.hf1(1,12), 13)


if __name__=='__main__':
    unittest.main()
