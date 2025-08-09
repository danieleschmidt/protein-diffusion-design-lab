"""
Minimal mock implementations for torch when not available.
"""

from typing import Any, Union, List, Tuple
import random

class MockTensor:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                self._shape = (len(data), len(data[0]))
            else:
                self._shape = (len(data),)
        else:
            self._shape = ()
        
    @property
    def shape(self):
        return self._shape
    
    def size(self, dim=None):
        return self._shape[dim] if dim is not None else self._shape
    
    def tolist(self):
        return self.data if isinstance(self.data, list) else [self.data]
    
    def cpu(self):
        return self
    
    def to(self, device=None, dtype=None):
        return self
    
    def unsqueeze(self, dim):
        return self
    
    def mean(self, dim=None):
        return MockTensor(0.5)
    
    def max(self, dim=None):
        return MockTensor(1.0), MockTensor(0)
    
    def argmax(self, dim=-1):
        return MockTensor([0] * max(1, self._shape[0] if self._shape else 1))
    
    def item(self):
        return 0.5
    
    def __getitem__(self, key):
        return MockTensor(0.5)
    
    def any(self):
        return MockTensor(True)
    
    @property
    def device(self):
        return MockDevice("cpu")

def tensor(data, dtype=None):
    return MockTensor(data, dtype)

def zeros(*shape, dtype=None):
    return MockTensor([0.0] * shape[0] if shape else 0.0, dtype)

def randn(*shape, dtype=None):
    return MockTensor([random.gauss(0, 1) for _ in range(shape[0])] if shape else random.gauss(0, 1), dtype)

def isnan(input):
    return MockTensor(False)

def isinf(input):
    return MockTensor(False)

class MockDevice:
    def __init__(self, device_str):
        self.type = device_str
    
    def __str__(self):
        return self.type

def device(device_str):
    return MockDevice(device_str)

class MockCuda:
    @staticmethod
    def is_available():
        return False
    
    @staticmethod
    def device_count():
        return 0

cuda = MockCuda()

class MockDtype:
    def __init__(self, name):
        self.name = name

float16 = MockDtype("float16")
float32 = MockDtype("float32")
long = MockDtype("long")

class MockModule:
    def __init__(self):
        self.training = False
    
    def parameters(self):
        return []
    
    def to(self, device=None, dtype=None):
        return self
    
    def train(self, mode=True):
        self.training = mode
        return self
    
    def eval(self):
        self.training = False
        return self

class MockLinear(MockModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
    
    def __call__(self, input):
        return MockTensor([0.5] * self.out_features)

class MockEmbedding(MockModule):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def __call__(self, input):
        return MockTensor([[0.5] * self.embedding_dim] * 10)

class MockNN:
    Module = MockModule
    Linear = MockLinear
    Embedding = MockEmbedding
    
    class LayerNorm(MockModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def __call__(self, input):
            return input
    
    class Dropout(MockModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def __call__(self, input):
            return input
    
    class ModuleList(MockModule):
        def __init__(self, modules=None):
            super().__init__()
            self.modules = modules or []
        def __iter__(self):
            return iter(self.modules)
    
    class Identity(MockModule):
        def __call__(self, input):
            return input

nn = MockNN()

class MockF:
    @staticmethod
    def mse_loss(input, target):
        return MockTensor(0.5)
    
    @staticmethod
    def softmax(input, dim=-1):
        return input
    
    @staticmethod
    def gelu(input):
        return input

F = MockF()

class no_grad:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def load(f, map_location=None):
    return {}

def save(obj, f):
    pass

# Alias for type hints
Tensor = MockTensor
dtype = MockDtype