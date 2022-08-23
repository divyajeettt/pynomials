# pynomials

## About pynomials

pynomials is a mathematical module providing Polynomials (in one variable, anmely `x`) as Python objects through a class called 'Polynomial'.

*Date of creation:* `May 17, 2021` \
*Date of first release on [PyPI](https://pypi.org/project/pynomials/):* `June 02, 2021`

Using this module, algebra can be performed easily. The module is enriched with help-text for a smooth user-experience. The module also provides functions to graph polynomials, which can be used to enhance visualization capabilities of students.

## About class `Polynomial`

To access the help-text of the class to know more about it, run:

```python
help(pynomials.Polynomial)
```

### Some features

- Polynomials can be added/subtracted to get the corresponding Polynomial
- Polynomials can be multiplied with a real number or another Polynomial 
- Polynomials can also be divided by other Polynomials to get the quotient/remainder of the divison, and to get a corresponding "Rational-Function"
- Polynomials can also be raised to integer powers, differentiated, and integrated

Users are advised to look at the class' help-text to learn about the available functionalities.

## Update History

### Updates (0.0.4)

Added two more methods to class `Polynomial`:
- `Polynomial.definite_derivative(p, n)`
    > returns the value of derivaitve of `p` at `x = n`
- `Polynomial.definite_integral(p, a, b)`
    > returns the value of integral of `p` from lower-limit `a` to upper-limit `b`

### Updates (0.0.5)

- Minor bug fixes
- Changes in division of Polynomials using // and % operators. Using these operators on Polynomials now strictly returns Polynomials (and not None).

## Footnotes

Additional objects available in the module (except class Polynommial) are not meant for use.

## Run

To use, execute:

```
pip install pynomials
```

Import the class `Polynomial` in your project, wherever needed, using:

```python
from pynomials import Polynomial
```
