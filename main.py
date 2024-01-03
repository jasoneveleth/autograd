import tracednp as jnp
from tracednp import J, Node, W, bv
from numpy import random, ones, allclose
import jax.numpy as jjnp
import jax

def show(expr):
    """Visualize the computational graph up to expr"""
    def _show(self):
        s = f"{self.id} [label=\"{self.f}:{self.val}\"]\n"
        for c in self.parents:
            s += f"{show(c)}\n"
            s += f"{c.id} -> {self.id}\n"
        return s
    return "digraph g {\nnode [shape = \"circle\"]\n" + _show(expr) + "\n}"

def grad(f):
    def ret(*args):
        # assume already nodes
        out = f(*args)

        g = Node('g', 1, [])
        while True:
            if len(out.parents) == 0:
                break
            g = J[out.f](out.parents[0], g)
            out = out.parents[0]
        return g

    return ret

x = Node('x', random.random(8), [])
# y = Node('y', random.random(10), [])
# print(show(x + y))

def f(x):
    return jnp.sum(jnp.sin(jnp.b(jnp.w(x))))
gf = grad(f)
assert(allclose(gf(x).val, jnp.cos(jnp.b(jnp.w(x))).val@W))

def g(x):
    return jnp.sum(jnp.w(x))
gg = grad(g)
assert(allclose(gg(x).val, ones((1, 10))@ W))

def jf(x):
    return jjnp.sum(jjnp.sin(W@x + bv))
gjf = jax.grad(jf)
print(gjf(x.val))
print(gf(x))
