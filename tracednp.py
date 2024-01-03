import numpy as np

class Node:
    next_id = 0
    def __init__(self, f, val, parents):
        self.f = f
        self.val = val
        self.parents = parents
        self.id = Node.next_id
        Node.next_id += 1

    def __neg__(self):
        return Node('u-', -self.val, [self])

    def __add__(self, other):
        return Node('+', self.val + other.val, [self, other])

    def __mul__(self, other):
        return Node('*', self.val * other.val, [self, other])

    def __str__(self):
        return f"{self.f}: {self.val}, {list(map(lambda x: x.f, self.parents))}"

    def __setattr__(self, name, value):
        if hasattr(self, name):
            current_value = self.__dict__[name]
            raise AttributeError(f"Cannot modify attribute 'self.{name}' from '{current_value}' to '{value}'. This class is immutable.")
        else:
            # If the attribute doesn't exist, allow the assignment
            super().__setattr__(name, value)


W = np.random.random((10, 8))
bv = np.random.random((10))

J = {}

def sin(x):
    return Node('sin', np.sin(x.val), [x])

def cos(x):
    return Node('cos', np.cos(x.val), [x])

def sum(x):
    return Node('sum', np.sum(x.val), [x])

def J_sum(x, g):
    return Node('dsum', np.ones_like(x.val) * g.val, [g])

def J_sin(x, g):
    return cos(x)*g

def J_cos(x, g):
    return sin(x)*g

def w(x):
    return Node('w', W @ x.val, [x])

def J_w(_, g):
    return Node('@', W.T @ g.val, [g])

def b(x):
    return Node('b', x.val + bv, [x])

def J_b(_, g):
    return g

def installJ(s, f):
    global J
    J[s] = f

installJ('sin', J_sin)
installJ('cos', J_cos)
installJ('w', J_w)
installJ('b', J_b)
installJ('sum', J_sum)
