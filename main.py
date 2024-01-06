import tracednp as jnp
from tracednp import J, Node, W, bv
from numpy import random, ones, allclose
import jax.numpy as jjnp
import jax

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

def g(x):
    return jnp.sum(jnp.w(x))
gg = grad(g)
# check that our gradient is right on simple function
assert(allclose(gg(x).val, ones((1, 10))@ W))

def f(x):
    return jnp.sum(jnp.sin(jnp.b(jnp.w(x))))
gf = grad(f)
# check that our gradient is the same as the gradient I computed for `f`
assert(allclose(gf(x).val, jnp.cos(jnp.b(jnp.w(x))).val@W))

def jf(x):
    return jjnp.sum(jjnp.sin(W@x + bv))
gjf = jax.grad(jf)
# check that JAX and my code agree
assert(allclose(gjf(x.val), gf(x).val))
