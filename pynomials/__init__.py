"""pynomials
Provides mathematical polynomials in one variable as python objects"""


from types import GeneratorType
from typping import Callable
import matplotlib.pyplot as plt
import numpy as np
import math


SUPERSCRIPTS: dict[str, str] = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

def to_superscript(n: str) -> str:
    return n.translate(SUPERSCRIPTS)


class Polynomial:
    """Polynomial(aₙ, aₙ₋₁, ..., a₂, a₁, a₀) ->
    aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₂x² + a₁x¹ + a₀x⁰
    represents a Real Polynomial with degree 'n' and given coefficients

    see help(__init__) for help on Polynomial creation
    Truth Value of Polynomials:
        bool(Polynomial(0)) == False"""


    def __init__(self, /, *coefficients: float) -> None:
        """initializes Polynomial instance
        last term given to the Polynomial() constructor is taken as the constant

        attrs of Polynomial:
            • self.degree = highest power of x
            • self.coeffs = (aₙ, aₙ₋₁, ..., a₂, a₁, a₀)
            • self.a0, self.a1, self.a2, ... = a₀, a₁, a₂, ...
        Note: degree of Zero Polynomial is None

        examples:
            • Polynomial(1, -3, 4) -> x² - 3x¹ + 4x⁰
            • Polynomial(1, 2) -> x¹ + 2x⁰
            • Polynomial(1, 2, 0) -> x² + 2x¹"""

        coeffs = []
        for coeff in coefficients:
            if not isinstance(coeff, int|float):
                raise TypeError(
                    "all coefficients of Polynomial must be 'int' / 'float'"
                )
            if str(abs(coeff)) in {"nan", "inf"}:
                raise ValueError(f"invalid coefficient for 'Polynomial': '{coeff}'")

            coeffs.append(int(coeff) if float(coeff).is_integer() else coeff)

        if not any(coeffs):
            coeffs = (0, )

        if len(coeffs) != 1 and not coeffs[0]:
            raise ValueError("leading term of the Polynomial cannot be zero")

        self.coeffs = tuple(coeffs)
        self.degree = len(self.coeffs) - 1

        if not self.degree and not self.coeffs[0]:
            self.degree = None

        for i, coeff in enumerate(reversed(self.coeffs)):
            setattr(self, f"a{i}", coeff)


    def __str__(self, /) -> str:
        """defines str(self)"""

        strings = []
        for i, coeff in enumerate(reversed(self.coeffs)):
            if not coeff:
                continue
            else:
                sign = "+ " if coeff >= 0 else "- "
                power = to_superscript(str(i))
                value = "" if abs(coeff) == 1 else abs(coeff)
                strings.append(f"{sign}{value}x{power}")

        if not strings:
            return "0x⁰"
        else:
            return " ".join(reversed(strings)).lstrip("+ ")


    def __repr__(self, /) -> str:
        """defines repr(self)"""

        if not self:
            return "Polynomial(0)"
        else:
            return f"Polynomial{self.coeffs}"


    def __bool__(self, /) -> bool:
        """defines bool(self)
        bool(Polynomial(0)) == False"""

        return not (
            math.isclose(self.coeffs[0], 0, abs_tol=1e-09)
            and len(self.coeffs) == 1
        )


    def __hash__(self, /) -> int:
        """defines hash(self)"""

        return hash(repr(self))


    def __getitem__(self, exponent: int) -> float:
        """defines indexing property for Polynomial
        returns the coefficient of xⁿ where n is the index (at degree n)
        example:
            p = Polynomial(1, -3, 0, 2)
            • p[0] -> 2       • p[2] -> -3
            • p[1] -> 0       • p[3] -> 1"""

        if not isinstance(exponent, int):
            raise TypeError("Polynomial indices must be 'int'")

        deg = 1 if self.degree is None else self.degree + 1

        if exponent in range(0, deg):
            return self.coeffs[self.degree-exponent]
        else:
            raise ValueError("Polynomial indices must be between (0 - degree)")


    def __call__(
            self, /, q: int|float|np.ndarray|'Polynomial'
        ) -> float|'Polynomial':
        """if q is 'int' / 'float' / 'numpy.ndarray', returns the value of the
        Polynomial Function at x = q, i.e., P(q) where P = self
        if q is 'Polynomial', returns the Polynomial obtained by composition
        of P with q, i.e. P(q(x)) where P = self"""

        if not isinstance(q, int|float|Polynomial|np.ndarray):
            raise TypeError(" ".join((
                "invalid argument type for 'q':",
                "'int', 'float', 'Polynomial' or 'numpy.ndarray'"
            )))

        total = Polynomial(0) if isinstance(q, Polynomial) else 0
        for i, a in enumerate(reversed(self.coeffs)):
            total += a * q**i
        return total


    def __contains__(self, point: tuple[float], /) -> bool:
        """defines membership property for Polynomial
        return True if point (x, y) is of the form (x, P(x)) where P = self
        i.e., if y is the value of P at x"""

        if not isinstance(point, tuple):
            raise TypeError(" ".join((
                "'in <Polynomial>' requires left operand as 'tuple' of floats",
                f"not {point.__class__.__name__ !r}"
            )))
        if len(point) != 2:
            raise ValueError("left operand must be of the form (x, y)")

        for num in point:
            if not isinstance(num, int|float):
                raise TypeError("left operand must be 'tuple' of floats")

        return point == (point[0], self(point[0]))


    def __neg__(self, /) -> 'Polynomial':
        """defines the negative of a Polynomial using unary '-' operator
        returns the Polynomial obtained after multiplying each of its
        coefficients by -1
        equivalent to self * -1"""

        return self * -1


    def __eq__(self, other: 'Polynomial', /) -> bool:
        """defines the equality of Polynomial Objects using '==' operator"""

        if type(self) is not type(other):
            return False
        else:
            return self.coeffs == other.coeffs


    def __add__(self, other: 'Polynomial', /) -> 'Polynomial':
        """add Polynomial Objects using '+' operator
        returns the result for (self + other)"""

        if not all({self, other}):
            return self if not other else other

        coeffs1 = self.coeffs if self.degree >= other.degree else other.coeffs
        coeffs2 = self.coeffs if coeffs1 != self.coeffs else other.coeffs

        while len(coeffs2) != len(coeffs1):
            coeffs2 = (0,) + coeffs2

        polynomial = [coeffs1[i] + coeffs2[i] for i in range(len(coeffs1))]

        if not any(polynomial):
            return Polynomial(0)
        else:
            while not polynomial[0]:
                polynomial.pop(0)
        return Polynomial.FromSequence(polynomial)


    def __sub__(self, other: 'Polynomial', /) -> 'Polynomial':
        """subtract Polynomial Objects using '-' operator
        returns the result for (self - other)"""

        return self + (-other)


    def __mul__(self, other: float|'Polynomial', /) -> 'Polynomial':
        """multiply Polynomial Objects using '*' operator
        other can also be a number by which all components of the Polynomial
        will be multipled
        returns the result for (self * other)"""

        if isinstance(other, int|float):
            return Polynomial.FromSequence(coeff*other for coeff in self.coeffs)
        else:
            coeffs, powers = [], []

        for i, a1 in enumerate(self.coeffs):
            for j, a2 in enumerate(other.coeffs):
                coeffs.append(a1 * a2)
                powers.append(self.degree - i + other.degree - j)

        polynomial = Polynomial(0)
        for coeff, power in zip(coeffs, powers):
            if not coeff:
                continue
            else:
                polynomial += Polynomial(coeff, *[0]*power)

        return polynomial

    __rmul__ = __mul__


    def __pow__(self, n: int, /) -> 'Polynomial':
        """returns the Polynomial obtained by raising it to the positive integer
        power n using '**' operator
        returns the result for (self ** n)"""

        if not isinstance(n, int):
            raise TypeError("** for Polynomial requires right operand as 'int'")
        if n < 0:
            raise ValueError(
                "Polynomials can only be raised to +ve integer powers"
            )

        polynomial = Polynomial(1)
        for _ in range(n):
            polynomial *= self
        return polynomial


    def __truediv__(self, other: 'Polynomial', /) -> Callable[[float], float]:
        """divide two Polynomial Objects using '/' operator and get a callable
        object, specifically a Rational Function which takes one argument (x)
        returns the result for (self / other)
        Note: cannot divide Polynomial by Polynomial(0)"""

        if not other:
            raise ZeroDivisionError("cannot divide Polynomial by Polynomial(0)")
        else:
            return (lambda x: self(x) / other(x))


    def __floordiv__(self, other: 'Polynomial', /) -> 'Polynomial':
        """divide Polynomial Objects using '//' operator and get the quotient
        returns the result for (self // other)
        i.e., returns the Quotient obtained when self is divided by other
        Note: cannot divide Polynomial by Polynomial(0)"""

        if not other:
            raise ZeroDivisionError("cannot divide Polynomial by Polynomial(0)")
        if other.degree > self.degree:
            return Polynomial(0)

        p1, p2, polynomial, power = self, other, Polynomial(0), 1
        while power > 0:
            coeff = p1.coeffs[0] / p2.coeffs[0]
            try:
                power = p1.degree - p2.degree
            except TypeError:
                break
            else:
                polynomial += (p := Polynomial(coeff, *[0]*power))
                p1 -= (p2 * p)

        return polynomial


    def __mod__(self, other: 'Polynomial', /) -> 'Polynomial':
        """divide Polynomial Objects using '%' operator and get the remainder
        returns the result for (self % other)
        i.e., returns the Remainder when self is divided by other
        Note: cannot divide Polynomial by Polynomial(0)"""

        if not other:
            raise ZeroDivisionError("cannot divide Polynomial by Polynomial(0)")
        if other.degree > self.degree:
            return self

        p1, p2, power = self, other, 1
        while power > 0:
            coeff = p1.coeffs[0] / p2.coeffs[0]
            power = p1.degree - p2.degree
            p1 -= (p2 * Polynomial(coeff, *[0]*power))

        return p1 if other.degree > p1.degree else p1 % other


    @classmethod
    def FromSequence(cls, sequence: list|tuple|GeneratorType, /) -> 'Polynomial':
        """creates a Polynomial from the given sequence / generator
        the sequence must contain / generator must yield only 'int' / 'float'"""

        if not isinstance(sequence, list|tuple|GeneratorType):
            raise TypeError(" ".join((
                "invalid sequence type for 'Polynomial':",
                "must be 'list', 'tuple' or 'generator'"
            )))
        if isinstance(sequence, GeneratorType):
            sequence = tuple(sequence)

        return cls(*sequence)


    @classmethod
    def FromRoots(cls, *roots: float) -> 'Polynomial':
        """creates a Polynomial of degree n whose roots are given as arguments,
        where n is the number of roots
        example:
            • FromRoots(a₁, a₂, ..., aₙ) -> (x - a₁)(x - a₂)...(x - aₙ)"""

        for root in roots:
            if not isinstance(root, int|float):
                raise TypeError(
                    "'FromRoots' expects all arguments as 'int' / 'float'"
                )
            if str(abs(root)) in {"nan", "inf"}:
                raise ValueError(f"invalid root for 'Polynomial': '{coeff}'")

        polynomial = Polynomial(1)
        for root in roots:
            polynomial *= Polynomial(1, -root)
        return polynomial


    @staticmethod
    def plot_polynomials(
            *polynomials: 'Polynomials', show_legend: bool|None = True
            limits: tuple[float]|None = (-100.0, 100.0),
        ) -> None:
        """plots all given Polynomials in the same graph
        show_legend is a boolean which displays legend on the plot if True
        limits should be a tuple containing min and max x limits (floats)"""

        if not isinstance(limits, tuple):
            raise TypeError("expected argument type for 'limits': 'tuple'")
        if len(limits) != 2:
            raise ValueError("'limits' must be a 'tuple' of two floats")

        for num in limits:
            if not isinstance(num, int|float):
                raise TypeError("'limits' must be a 'tuple' of two floats")

        if not polynomials:
            return

        colors = [
            "red", "orange", "green", "blue", "purple",
            "cyan", "magenta", "yellow", "brown", "gray",
        ] * len(polynomials)

        plt.axhline(y=0, color="black", linewidth=2.5)
        plt.axvline(x=0, color="black", linewidth=2.5)
        plt.plot(0, 0, color="black", marker="o")

        x = np.arange(limits[0], limits[1], 0.0078125)
        for i, polynomial in enumerate(polynomials):
            if polynomials.index(polynomial) == i:
                plt.plot(
                    x, polynomial(x), linewidth=2, label=polynomial,
                    color=(colors[i] if polynomial else "black")
                )

        if show_legend:
            plt.legend()

        plt.grid(True)
        plt.show()


    def plot(self, /, *, limits: tuple[float]|None = (-100.0, 100.0)) -> None:
        """plots the Polynomial on a graph with given min and max x limits
        (floats) and displays it"""

        Polynomial.plot_polynomials(self, limits=limits)


    def derivative(self, /) -> 'Polynomial':
        """returns the Polynomial obtained by differentiating self"""

        if not self:
            return Polynomial(0)

        coeffs = list(self.coeffs)
        for i, power in enumerate(range(self.degree, -1, -1)):
            coeffs[i] *= power
        coeffs.pop()

        return Polynomial.FromSequence(coeffs)


    def integral(self, /, *, c: float|None = 0) -> 'Polynomial':
        """returns the Polynomial obtained by integrating self at integrating
        constant = c"""

        if not isinstance(c, int|float):
            raise TypeError("expected argument type(s) for c: 'int' / 'float'")

        coeffs = list(self.coeffs)
        for i, power in enumerate(range(self.degree, -1, -1)):
            coeffs[i] /= (power + 1)
        coeffs.append(c)

        return Polynomial.FromSequence(coeffs)


    def nth_derivative(self, n: int|None = 1, /) -> 'Polynomial':
        """returns the Polynomial obtained by differentiating self n times"""

        if not isinstance(n, int):
            raise TypeError("invalid argument type for 'n': must be 'int'")
        if n < 0:
            raise ValueError("'n' must be positive 'int'")

        polynomial = self
        for _ in range(n):
            polynomial = polynomial.derivative()

        return polynomial


    def definite_derivative(self, x: float, /) -> float:
        """returns the value of the derivative of self at x value = x"""

        return self.derivative()(x)


    def definite_integral(self, x1:  float, x2: float, /) -> float:
        """returns the value of the integral of self from x = x1 to x = x2
        taking x1 to be the lower limit"""

        return self.integral()(x2) - self.integral()(x1)