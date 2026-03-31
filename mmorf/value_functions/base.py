from typing import Union
import copy
class ValueFn(object):
    def __init__(self, *args):
        self.args = args
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []
    
    def __mul__(self, other):
        if isinstance(other, int):
            return self * float(other)
        elif isinstance(other, float):
            mycopy = copy.deepcopy(self)
            # scale both the coeff and the intercept so scalar multiplication
            # follows algebraic distributive semantics: (f + k) * s = f*s + k*s
            mycopy.coeff = self.coeff * other
            mycopy.intercept = self.intercept * other
            mycopy.children = self.children.copy()
            return mycopy
        elif isinstance(other, ValueFn):
            parent = ValueFnProduct(self, other)
            return parent
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
        
    def __rmul__(self, other): 
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, int):
            return self / float(other)
        elif isinstance(other, float):
            mycopy = copy.deepcopy(self)
            # divide both the coeff and the intercept so scalar division
            # follows algebraic semantics: (f + k) / s = f/s + k/s
            mycopy.coeff = self.coeff / other
            mycopy.intercept = self.intercept / other
            mycopy.children = self.children.copy()
            return mycopy
        elif isinstance(other, ValueFn):
            parent = ValueFnQuotient(self, other)
            return parent
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        
    def __rtruediv__(self, other):
        parent = ValueFnQuotient(other, self)
        return parent
        
    def __add__(self, other):
        if isinstance(other, ValueFn):
            parent = ValueFn()
            parent.children.append(self)
            parent.children.append(other)
            return parent
        elif isinstance(other, int):
            return self + float(other)
        elif isinstance(other, float):
            mycopy = copy.deepcopy(self)
            mycopy.coeff = self.coeff
            mycopy.intercept = self.intercept + other
            mycopy.children = self.children.copy()
            return mycopy
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        if isinstance(other, ValueFn):
            parent = ValueFn()
            parent.children.append(self)
            parent.children.append(other * -1.0)
            return parent
        elif isinstance(other, int):
            return self - float(other)
        elif isinstance(other, float):
            mycopy = copy.deepcopy(self)
            mycopy.coeff = self.coeff
            mycopy.intercept = self.intercept - other
            mycopy.children = self.children.copy()
            return mycopy
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __rsub__(self, other):
        return -1.0 * self + other
    
    def __neg__(self):
        return -1.0 * self
    
    def __call__(self, **kwargs):
        return self.intercept + self.coeff * sum(child(**kwargs) for child in self.children)

    def __repr__(self):
        return f"{self.coeff} * ("+ " + ".join([repr(c) for c in self.children]) + f" + {self.intercept})"

class ValueFnProduct(ValueFn):
    def __init__(self, lhs: ValueFn, rhs: ValueFn, *args):
        super().__init__(*args)
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, **kwargs):
        return self.coeff * (self.lhs(**kwargs) * self.rhs(**kwargs)) + self.intercept
    
    def __repr__(self):
        return f"({repr(self.lhs)}) * ({repr(self.rhs)})"
    
class ValueFnQuotient(ValueFn):
    def __init__(self, lhs: Union[ValueFn, int, float], rhs: Union[ValueFn, float, int], *args):
        super().__init__(*args)
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, **kwargs):
        # Evaluate the underlying division, taking into account that lhs or rhs may be
        # plain numbers or ValueFn instances. Always apply this object's coeff and
        # intercept to the final numeric result.
        if isinstance(self.lhs, (int, float)):
            # lhs is a number, rhs should be evaluated if it's a ValueFn
            rhs_val = self.rhs(**kwargs) if isinstance(self.rhs, ValueFn) else float(self.rhs)
            result = float(self.lhs) / rhs_val
        elif isinstance(self.rhs, (int, float)):
            # rhs is a number, lhs should be evaluated if it's a ValueFn
            lhs_val = self.lhs(**kwargs) if isinstance(self.lhs, ValueFn) else float(self.lhs)
            result = lhs_val / float(self.rhs)
        else:
            result = self.lhs(**kwargs) / self.rhs(**kwargs)
        return self.coeff * result + self.intercept
    
    def __repr__(self):
        return f"({repr(self.lhs)}) / ({repr(self.rhs)})"

class ReactionLevelWrapper(ValueFn):
    """
    A value function that operates at the reaction level, and is summed up over all reactions before being combined with other value functions.
    """
    def __init__(self,*args):
        super().__init__(*args)
        self.child = self.child_type()(*args)

    def child_type(self):
        return NotImplementedError("ReactionLevelWrapper must have a child ValueFn type defined.")

    def get_value(self, reaction=None, node=None, **kwargs):
        reaction_value = 0.0
        if node is not None and hasattr(node, "reaction"):
            assert reaction == node.reaction, "ReactionLevelWrapper called with mismatched reaction and node"
            reaction = node.reaction
        if reaction is not None:
            kwargs["v_orig"] = 0
            reaction_value = self.child(reaction=reaction, node=node, **kwargs)
        if node is not None and node.parent is not None:
            return reaction_value + self.get_value(reaction = node.parent.reaction, node=node.parent, **kwargs)
        return reaction_value
    
    def __call__(self, reaction=None, node=None, **kwargs):
        reaction_value = self.get_value(reaction=reaction, node=node, **kwargs)
        return self.intercept + self.coeff * reaction_value

    def __repr__(self):
        return f"{self.coeff} * ({self.child_type().__name__}()) + {self.intercept}"
        