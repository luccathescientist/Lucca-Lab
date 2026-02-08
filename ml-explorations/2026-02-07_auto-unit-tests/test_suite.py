import unittest

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class SimpleRigModel:
    def __init__(self, name):
        self.name = name
    
    def get_status(self):
        return f"Model {self.name} is online"

class TestSimpleRigFunctions(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(-1, -1), -2)

    def test_subtract(self):
        self.assertEqual(subtract(10, 5), 5)
        self.assertEqual(subtract(-1, -1), 0)
        self.assertEqual(subtract(5, 10), -5)

    def test_multiply(self):
        self.assertEqual(multiply(3, 7), 21)
        self.assertEqual(multiply(-1, 5), -5)
        self.assertEqual(multiply(0, 100), 0)

    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(-10, 2), -5)
        with self.assertRaises(ValueError):
            divide(10, 0)

class TestSimpleRigModel(unittest.TestCase):

    def test_init(self):
        model = SimpleRigModel("TestRig")
        self.assertEqual(model.name, "TestRig")

    def test_get_status(self):
        model = SimpleRigModel("Alpha")
        self.assertEqual(model.get_status(), "Model Alpha is online")

if __name__ == '__main__':
    unittest.main()
