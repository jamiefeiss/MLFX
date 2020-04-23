import unittest

class TestClass(unittest.TestCase):
    def test_method(self):
        self.list = []
        self.list.append('str')
        self.assertTrue(self.list)

if __name__ == '__main__':
    unittest.main()