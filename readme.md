# PyToSolver

A generic framework for writing linear optimization problems in Python.

**The solver's Python package must be installed for this framework to work.**

If this package was useful in any way, please let me know.

### Installation

Installing from source

```
git clone https://github.com/guifcoelho/pytosolver
pip install ./pytosolver
```

Each specific solver package must also be installed. The current supported solvers are:

- [HiGHS](https://github.com/ERGO-Code/HiGHS) (the default solver, already comes with PyToSolver):

```
pip install highspy==1.7.2
```

- [Xpress](https://pypi.org/project/xpress/):

```
pip install xpress
```
