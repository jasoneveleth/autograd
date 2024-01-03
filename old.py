import sys
from enum import Enum

# we give every expr a unique character for when we output graphviz code
idx = ord('a') - 1
def expr2op_name(e):
    op_name = '+' if e.type == Type.SUM else '*'
    num = ord(e.name) - (ord('a') - 1)
    return f"{op_name}_{num}"

class Type(Enum):
    VAR = 0
    SUM = 1
    PROD = 2

class Expr:
    def __init__(self, val, thetype=Type.VAR, lhs=None, rhs=None, varname=None):
        if thetype is Type.VAR:
            assert(lhs is None)
            assert(rhs is None)
            assert(varname is not None)
        self.type = thetype
        self.val = val
        self.lhs = lhs
        self.rhs = rhs
        self.varname = varname

        global idx
        self.name = chr(idx := idx + 1)

    def __add__(self, a):
        return Expr(self.val + a.val, thetype=Type.SUM, lhs=self, rhs=a)

    def __mul__(self, a):
        return Expr(self.val * a.val, thetype=Type.PROD, lhs=self, rhs=a)

    # def __eq__(self, a):
    #     return self.type == a.type and self.val == a.val and self.lhs == a.lhs and self.rhs == a.rhs and self.varname == a.varname and self.name == self.name

    def __str__(self):
        s = ""
        if self.type is Type.VAR:
            s += f"{self.name} [label=\"{self.varname}={self.val}\"]"
        else:
            if self.lhs is None or self.rhs is None: raise RuntimeError("bad")
            s += f"{self.name} [label=\"{expr2op_name(self)}: {self.val}\"]\n"
            s += str(self.lhs) + "\n"
            s += str(self.rhs) + "\n"
            s += f"{self.lhs.name} -> {self.name}\n"
            s += f"{self.rhs.name} -> {self.name}"
        return s

    def __setattr__(self, name, value):
        if hasattr(self, name):
            current_value = self.__dict__[name]
            raise AttributeError(f"Cannot modify attribute 'self.{name}' from '{current_value}' to '{value}'. This class is immutable.")
        else:
            # If the attribute doesn't exist, allow the assignment
            super().__setattr__(name, value)

def show(expr):
    """Visualize the computational graph up to expr"""
    return """digraph g {
node [shape = "circle"]
""" + str(expr) + "\n}"

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]

# WONG: don't know what is returned, really :(
# @Memoize
# def grad(expr, verbose=False):
#     if verbose:
#         print("hi")
#     if expr.type is Type.VAR:
#         return expr
#     elif expr.type is Type.SUM:
#         if expr.lhs is None or expr.rhs is None: raise RuntimeError("bad")
#         return grad(expr.lhs, verbose) + grad(expr.rhs, verbose)
#     elif expr.type is Type.PROD:
#         if expr.lhs is None or expr.rhs is None: raise RuntimeError("bad")
#         return expr.rhs * grad(expr.lhs, verbose) + expr.lhs * grad(expr.rhs, verbose)

# WRONG: This calculates df/dx + df/dy
@Memoize
def grad(expr, verbose=False):
    if verbose:
        print("hi")
    if expr.type is Type.VAR:
        return 1
    elif expr.type is Type.SUM:
        if expr.lhs is None or expr.rhs is None: raise RuntimeError("bad")
        return grad(expr.lhs, verbose) + grad(expr.rhs, verbose)
    elif expr.type is Type.PROD:
        if expr.lhs is None or expr.rhs is None: raise RuntimeError("bad")
        return expr.rhs.val * grad(expr.lhs, verbose) + expr.lhs.val * grad(expr.rhs, verbose)

x = Expr(2,varname='x')
y = Expr(3,varname='y')
e = x*x + x*y
grad = grad(e)
print(grad)
# print(show(grad))
# print(show(e))
# print('grad', expr2op_name(grad), file=sys.stderr)
# print('e', expr2op_name(e), file=sys.stderr)

