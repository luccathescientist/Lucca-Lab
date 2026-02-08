import os
import sys

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

if __name__ == "__main__":
    print(add(5, 3))
