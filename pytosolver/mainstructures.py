from dataclasses import dataclass, field
from typing import Optional, Any
import math

from pytosolver.enums import ConstraintSign, VarType


@dataclass
class LinearExpression:
    "A wrapper for general expressions"
    elements: dict['Variable', float] = field(default_factory=dict, init=False)
    constant: float = field(default=0, init=False)

    def __len__(self):
        return len(self.elements)

    def get_sorted_vars(self):
        return sorted(list(self.elements.keys()), key=lambda el: el.get_name())

    def to_str(self):
        expr_str = ""
        for i, var in enumerate(self.get_sorted_vars()):
            coef = self.elements[var]
            if coef != 0:
                if i > 0 and coef > 0:
                    expr_str += " +"
                if i > 0:
                    expr_str += " "
                expr_str += f"{coef}*{var.get_name()}"
        if self.constant != 0:
            if self.constant > 0:
                expr_str += " +"
            expr_str += f" {self.constant}"
        return expr_str

    def __str__(self):
        return self.to_str()

    def copy(self):
        "Returns a copy of the linear expression"
        new_expr = LinearExpression()
        new_expr.elements = self.elements.copy()
        new_expr.constant = self.constant
        return new_expr

    def _add_expression(self, expr: "LinearExpression", addition: bool = True):
        sign = 1 if addition else -1
        self.constant += sign * expr.constant
        for key, val in expr.elements.items():
            self.elements[key] = self.elements.get(key, 0) + sign * val
        return self

    def _add_var(self, var: "Variable", addition: bool = True):
        sign = 1 if addition else -1
        self.elements[var] = self.elements.get(var, 0) + sign
        return self

    def _add_constant(self, val: float | int, addition: bool = True):
        sign = 1 if addition else -1
        self.constant += val * sign
        return self

    def _add(self, other, addition: bool = True):
        if isinstance(other, LinearExpression):
            return self._add_expression(other, addition)
        if isinstance(other, Variable):
            return self._add_var(other, addition)
        return self._add_constant(other, addition)

    def __add__(self, other: 'LinearExpression | Variable | float | int'):
        return self.copy()._add(other)

    def __radd__(self, other: 'LinearExpression | Variable | float | int'):
        return self + other

    def __iadd__(self, other: 'LinearExpression | Variable | float | int'):
        return self._add(other)

    def __sub__(self, other: 'LinearExpression | Variable | float | int'):
        return self.copy()._add(other, False)

    def __rsub__(self, other: 'LinearExpression | Variable | float | int'):
        return self - other

    def __isub__(self, other: 'LinearExpression | Variable | float | int'):
        return self._add(other, False)

    def _multiplication(self, coef: float | int, multiplication: bool = True):
        coef = float(coef)
        self.elements = {
            key: val * (coef if multiplication else 1/coef)
            for key, val in self.elements.items()
        }
        self.constant *= (coef if multiplication else 1/coef)
        return self

    def __mul__(self, coef: float | int):
        return self.copy()._multiplication(coef)

    def __rmul__(self, coef: float | int):
        return self * coef

    def __imul__(self, coef: float | int):
        return self._multiplication(coef)

    def __truediv__(self, coef: float | int):
        return self.copy()._multiplication(coef, False)

    def __rtruediv__(self, coef: float | int):
        return self/coef

    def __itruediv__(self, coef: float | int):
        return self._multiplication(coef, False)

    def __eq__(self, rhs: 'LinearExpression | Variable | float | int'):
        return LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.EQ))

    def __le__(self, rhs: 'LinearExpression | Variable | float | int'):
        return LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.LEQ))

    def __ge__(self, rhs: 'LinearExpression | Variable | float | int'):
        return LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.GEQ))

    def __neg__(self):
        return LinearExpression() - self

    def __pos__(self):
        return LinearExpression() + self

class LinearConstraint:
    "The linear constraint class"

    expression: 'LinearExpression'
    sign: ConstraintSign
    name: Optional[str] = None
    row: Optional[int] = None
    dual: Optional[float] = None

    _default_name: Optional[str] = None
    _hash: int = field(init=False)

    def __init__(self,
                 constr: 'LinearConstraint | tuple[LinearExpression, ConstraintSign]',
                 name: Optional[str] = None):
        if isinstance(constr, LinearConstraint):
            self.expression = constr.expression
            self.sign = constr.sign
        elif isinstance(constr, tuple):
            self.expression, self.sign = constr
        else:
            raise TypeError(f"The constraint expression must be an inequality, not `{constr}`.")

        self.name = name
        self._hash = id(self)

    def __hash__(self):
        if self._hash is None:
            raise Exception("The hash string of the 'LinearConstraint' object was not set.")
        return self._hash

    def get_name(self):
        return self.name or self.default_name or f"__constr{self.__hash__()}"

    def to_str(self):
        constr_str = self.expression.to_str() + f" {self.sign.value} 0"
        return f"{self.get_name()}: {constr_str}"

    def __str__(self):
        return self.to_str()

    @property
    def default_name(self):
        if self._default_name is None:
            raise Exception(
                "The default name of this constraint was not set. "
                "Add this constraint to the problem and run `problem.update()`"
            )
        return self._default_name

    @property
    def is_empty(self):
        return len(self.expression) == 0

    def set_row(self, row: int):
        self.row = row
        self.set_default_name()

    def clear_model_data(self):
        self.row = None
        self.dual = None
        self._default_name = None

    def set_default_name(self, row: Optional[int] = None):
        "Sets a default name to be provided to the solver."
        if row is None:
            row = self.row
        self._default_name = f"__constr{row}"
        return self

    def clear(self):
        "Clears all values from the constraint."
        self._default_name = None
        self.row = None
        self.dual = None

@dataclass
class Variable:
    "The decicion variable"

    name: Optional[str] = field(default=None)
    vartype: VarType = field(default=VarType.CNT)
    lowerbound: Optional[float] = field(default=None)
    upperbound: Optional[float] = field(default=None)
    hotstart: float | None = field(default=None, repr=False)
    target: float | None = field(default=None, repr=False)

    column: int | None = field(default=None, init=False)
    value: float | None = field(default=None, init=False)

    _default_name: str | None = field(default=None, init=False, repr=False)
    _hash: int = field(init=False)

    def __post_init__(self):
        self._hash = id(self)
        if self.vartype == VarType.BIN:
            self.lowerbound = 0
            self.upperbound = 1

        if self.hotstart is not None:
            self._validate_value(self.hotstart, 'hotstart')

        if self.target is not None:
            self._validate_value(self.target, 'target')

    def _validate_value(self, value: float, type) -> None:
        lb = 0 if self.vartype == VarType.BIN else self.lowerbound
        ub = 1 if self.vartype == VarType.BIN else self.upperbound
        if value < (lb or -math.inf) or value > (ub or math.inf):
            raise ValueError(
                f"The {type} for the decision variable is not within its bounds: "
                f"{value} < {lb or 'inf'}" if value < (lb or -math.inf) else f"{value} > {ub or 'inf'}"
            )

    @property
    def default_name(self):
        if self._default_name is None:
            raise Exception("The default name of this variable was not set.")
        return self._default_name

    def __hash__(self):
        if self._hash is None:
            raise Exception("The hash string of the 'Variable' object was not set.")
        return self._hash

    def get_name(self):
        return self.name or self._default_name or f"__var{self.__hash__()}"

    def to_str(self):
        name = self.get_name()
        var_str = f"{name}"
        bounds = (
            f"{self.lowerbound if self.lowerbound is not None else 'inf'} "
            f"<= {name} "
            f"<= {self.upperbound if self.upperbound is not None else 'inf'}"
        )
        if self.vartype == VarType.BIN:
            var_str += " in {0, 1}"
        elif self.vartype == VarType.INT:
            var_str += f" in I, {bounds}"
        elif self.vartype == VarType.CNT:
            var_str += f" in R, {bounds}"
        else:
            raise Exception("Unknown variable type.")
        return var_str

    def __str__(self):
        return self.to_str()

    def to_linexpr(self):
        "Transforms the variable into a linear expression"
        return LinearExpression() + self

    def __add__(self, other: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() + other

    def __radd__(self, other: 'LinearExpression | Variable | float | int'):
        return self + other

    def __sub__(self, other: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() - other

    def __rsub__(self, other: 'LinearExpression | Variable | float | int'):
        return self - other

    def __mul__(self, val: float | int):
        return self.to_linexpr() * float(val)

    def __rmul__(self, val: float | int):
        return self * float(val)

    def __truediv__(self, val: float | int):
        if not isinstance(val, (float, int)):
            raise ValueError("A variable can only be divided by a number")
        return self.to_linexpr() / float(val)

    def __rtruediv__(self, val):
        if not isinstance(val, (float, int)):
            raise ValueError("A variable can only be divided by a number")
        return self.to_linexpr() / float(val)

    def __eq__(self, rhs: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() == rhs

    def __le__(self, rhs: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() <= rhs

    def __ge__(self, rhs: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() >= rhs

    def __neg__(self):
        return LinearExpression() - self

    def __pos__(self):
        return LinearExpression() + self

    def clear_model_data(self):
        self.column = None
        self.value = None
        self._default_name = None

    def set_column(self, column: int):
        self.column = column
        self.set_default_name()

    def set_default_name(self, column: Optional[int] = None):
        "Sets a default name to be provided to the solver."
        if column is None:
            column = self.column
        self._default_name = f"__var{column}"
        return self

    def set_hotstart(self, value: float):
        "Sets the value to be used as initial solution on the solver."
        value = float(value)
        self._validate_value(value, 'hotstart')
        self.hotstart = float(value)
        return self

    def set_target(self, value: float):
        "Sets the target value for the decision variable."
        value = float(value)
        self._validate_value(value, 'target')
        self.target = value
        return self

    def clear(self):
        "Clears all values from the variable."
        self._default_name = None
        self.value = None
        self.column = None

@dataclass
class ObjectiveFunction:
    expression: "LinearExpression" = field(default_factory=LinearExpression)
    is_minimization: bool = True
    name: str | None = field(default=None)
    options: dict[str, Any] = field(default_factory=dict, hash=False, repr=False)
    value: float | None = field(default=None, init=False, hash=False)
    multi_obj_next_iter_slack: float = 0.

    _hash: int = field(init=False)

    def __hash__(self):
        if self._hash is None:
            raise Exception("The hash string of the 'ObjectiveFunction' object was not set.")
        return self._hash

    def __post_init__(self):
        self._hash = id(self)
        self.expression += LinearExpression()
        if not (0. <= self.multi_obj_next_iter_slack <= 1.):
            raise ValueError(f"The slack for the objective function '{self.get_name()}' must be between 0 and 1.")

    def get_name(self):
        return self.name or f"__obj{self.__hash__()}"

    def to_str(self):
        sense_str = "Minimize" if self.is_minimization else "Maximize"
        return f"{self.get_name()} | {sense_str}: {self.expression.to_str()}"

    def __str__(self):
        return self.to_str()
