"""
Enhanced mock implementations for torch when not available.
Now uses the enhanced mock framework for better compatibility.
"""

from typing import Any, Union, List, Tuple
import random

# Import enhanced mock framework
try:
    from .enhanced_mock_framework import (
        EnhancedMockTensor, EnhancedMockEmbedding, EnhancedMockLinear,
        EnhancedMockModule, EnhancedMockOptimizer, enhanced_mock_torch
    )
    ENHANCED_FRAMEWORK_AVAILABLE = True
except ImportError:
    ENHANCED_FRAMEWORK_AVAILABLE = False

class MockTensor:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                if data[0] and isinstance(data[0][0], list):
                    # 3D tensor
                    self._shape = (len(data), len(data[0]), len(data[0][0]))
                else:
                    # 2D tensor
                    self._shape = (len(data), len(data[0]) if data[0] else 0)
            else:
                # 1D tensor
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
    
    def __setitem__(self, key, value):
        pass  # Mock assignment
    
    def any(self):
        return MockTensor(True)
    
    def repeat(self, *sizes):
        return self
    
    def expand(self, *sizes):
        return self
    
    def transpose(self, dim0, dim1):
        return self
    
    def contiguous(self):
        return self
    
    def view(self, *shape):
        return self
    
    def flatten(self):
        return self
    
    def squeeze(self, dim=None):
        return self
    
    def type_as(self, other):
        return self
    
    def float(self):
        return self
    
    def long(self):
        return self
    
    def int(self):
        return self
    
    def __mul__(self, other):
        return self
    
    def __add__(self, other):
        return self
    
    def __sub__(self, other):
        return self
    
    def __truediv__(self, other):
        return self
    
    def __rtruediv__(self, other):
        return MockTensor(0.5)
    
    def __pow__(self, other):
        return self
    
    def __rpow__(self, other):
        return MockTensor(1.0)
    
    def __rsub__(self, other):
        return MockTensor(0.5)
    
    def __rmul__(self, other):
        return self
    
    def __radd__(self, other):
        return self
    
    @property
    def device(self):
        return MockDevice("cpu")

def tensor(data, dtype=None):
    return MockTensor(data, dtype)

def zeros(*shape, dtype=None):
    return MockTensor([0.0] * shape[0] if shape else 0.0, dtype)

def randn(*shape, dtype=None, device=None):
    if len(shape) == 1:
        return MockTensor([random.gauss(0, 1) for _ in range(shape[0])], dtype)
    elif len(shape) == 2:
        return MockTensor([[random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])], dtype)
    elif len(shape) == 3:
        return MockTensor([[[random.gauss(0, 1) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])], dtype)
    return MockTensor(random.gauss(0, 1), dtype)

def isnan(input):
    return MockTensor(False)

def isinf(input):
    return MockTensor(False)

def arange(start, end=None, step=1, dtype=None):
    if end is None:
        end = start
        start = 0
    return MockTensor(list(range(int(start), int(end), int(step))), dtype)

def linspace(start, end, steps):
    if steps == 1:
        return MockTensor([start])
    step = (end - start) / (steps - 1)
    return MockTensor([start + i * step for i in range(steps)])

def randint(low, high, size, device=None, dtype=None):
    if isinstance(size, tuple):
        return MockTensor([random.randint(low, high-1) for _ in range(size[0])], dtype)
    return MockTensor([random.randint(low, high-1) for _ in range(size)], dtype)

def full(size, fill_value, device=None, dtype=None):
    if isinstance(size, tuple):
        return MockTensor([fill_value] * size[0], dtype)
    return MockTensor([fill_value] * size, dtype)

def cat(tensors, dim=0):
    return tensors[0] if tensors else MockTensor([])

def matmul(tensor1, tensor2):
    return MockTensor([[0.5] * 10] * 10)

def outer(tensor1, tensor2):
    return MockTensor([[0.5] * 10] * 10)

def sqrt(input):
    return input

def exp(input):
    return input

def cos(input):
    return input

def sin(input):
    return input

def clamp(input, min_val=None, max_val=None):
    return input

def cumprod(input, dim):
    return input

def ones(*shape, dtype=None):
    if len(shape) == 1:
        return MockTensor([1.0] * shape[0], dtype)
    elif len(shape) == 2:
        return MockTensor([[1.0] * shape[1]] * shape[0], dtype)
    return MockTensor(1.0, dtype)

class MockFinfo:
    def __init__(self, dtype):
        self.dtype = dtype
        self.min = -1e10
        self.max = 1e10

def finfo(dtype):
    return MockFinfo(dtype)

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
float = MockDtype("float")
long = MockDtype("long")
int64 = MockDtype("int64")
bool = MockDtype("bool")

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
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def apply(self, fn):
        return self
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
    
    def __call__(self, *args, **kwargs):
        """Mock callable behavior - routes to forward method if it exists."""
        if hasattr(self, 'forward'):
            return self.forward(*args, **kwargs)
        return MockTensor([0.5] * 10)

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
        # Return tensor with shape [batch_size, seq_len, embedding_dim]
        if hasattr(input, 'shape') and len(input.shape) >= 2:
            batch_size, seq_len = input.shape[0], input.shape[1]
        else:
            batch_size, seq_len = 1, 10  # Default fallback
        
        return MockTensor([[[0.5] * self.embedding_dim for _ in range(seq_len)] for _ in range(batch_size)])

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
    
    @staticmethod
    def relu(input):
        return input
    
    @staticmethod
    def silu(input):
        return input
    
    @staticmethod
    def pad(input, pad, mode='constant', value=0):
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

# Enhanced mock compatibility layer
if ENHANCED_FRAMEWORK_AVAILABLE:
    # Use enhanced mocks for better compatibility
    def get_enhanced_tensor(*args, **kwargs):
        return EnhancedMockTensor(*args, **kwargs)
    
    def get_enhanced_embedding(*args, **kwargs):
        return EnhancedMockEmbedding(*args, **kwargs)
    
    def get_enhanced_linear(*args, **kwargs):
        return EnhancedMockLinear(*args, **kwargs)
    
    # Override classes to use enhanced versions
    class EnhancedMockEmbeddingWrapper(MockEmbedding):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__(num_embeddings, embedding_dim)
            self._enhanced = EnhancedMockEmbedding(num_embeddings, embedding_dim)
            # Add weight attribute for compatibility
            self.weight = self._enhanced.weight
        
        def __call__(self, input):
            return self._enhanced(input)
    
    # Use enhanced embedding for better compatibility
    MockEmbedding = EnhancedMockEmbeddingWrapper
    nn.Embedding = EnhancedMockEmbeddingWrapper