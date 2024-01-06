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

def show(node):
    """Visualize the computational from `node`, produces output that
       can be fed directly into graphviz `dot` [here][online graphviz]

       [oneline graphviz]: https://dreampuf.github.io/GraphvizOnline/#digraph%20G%20%7B%0A%0A%20%20subgraph%20cluster_0%20%7B%0A%20%20%20%20style%3Dfilled%3B%0A%20%20%20%20color%3Dlightgrey%3B%0A%20%20%20%20node%20%5Bstyle%3Dfilled%2Ccolor%3Dwhite%5D%3B%0A%20%20%20%20a0%20-%3E%20a1%20-%3E%20a2%20-%3E%20a3%3B%0A%20%20%20%20label%20%3D%20%22process%20%231%22%3B%0A%20%20%7D%0A%0A%20%20subgraph%20cluster_1%20%7B%0A%20%20%20%20node%20%5Bstyle%3Dfilled%5D%3B%0A%20%20%20%20b0%20-%3E%20b1%20-%3E%20b2%20-%3E%20b3%3B%0A%20%20%20%20label%20%3D%20%22process%20%232%22%3B%0A%20%20%20%20color%3Dblue%0A%20%20%7D%0A%20%20start%20-%3E%20a0%3B%0A%20%20start%20-%3E%20b0%3B%0A%20%20a1%20-%3E%20b3%3B%0A%20%20b2%20-%3E%20a3%3B%0A%20%20a3%20-%3E%20a0%3B%0A%20%20a3%20-%3E%20end%3B%0A%20%20b3%20-%3E%20end%3B%0A%0A%20%20start%20%5Bshape%3DMdiamond%5D%3B%0A%20%20end%20%5Bshape%3DMsquare%5D%3B%0A%7D
       """
    def _show(self):
        s = f"{self.id} [label=\"{self.f}:{self.val}\"]\n"
        for c in self.parents:
            s += f"{show(c)}\n"
            s += f"{c.id} -> {self.id}\n"
        return s
    return "digraph g {\nnode [shape = \"circle\"]\n" + _show(node) + "\n}"

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
