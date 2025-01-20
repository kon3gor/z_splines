import numpy as np
from typing import List
import sympy as sym
from sympy import Piecewise

from .types import Function
from .helpers import factorial


def vander_matrix(m: int) -> List[List[float]]:
    """
    Generate Vandermonde matrix. See https://en.wikipedia.org/wiki/Vandermonde_matrix for details

    Parameters:
        m (int): m parameter of the matrix. Final matrix will be of size 2*m-1

    Returns:
        List[List[float]]: resulting Vandermonde matrix

    """
    tmp_matrix = np.zeros(shape=(2*m-1, 2*m-1))
    for l in range(1, 2*m):
        for p in range(1, 2*m):
            tmp_matrix[l-1, p-1] = (-(m-1) + (l-1))**(p-1)

    return tmp_matrix


def normalization_matrix(m: int) -> List[List[float]]:
    """

    Generates diogonal matrix of the following form: diag(1/0!, 1/1!, 1/2! ... ) 

    Parameters:
        m (int): m parameter of the matrix. Final matrix will be of size 2*m-1 

    Returns:
        List[List[float]]: resulting diagoanl matrix

    """
    diagonal_elements = [1 / factorial(l - 1) for l in range(1, 2*m)]
    D = np.diag(diagonal_elements)
    return D


def finite_difference_matrix(m: int) -> List[List[float]]:
    """

    Generate fintie difference matrix. 

    Parameters:
        m (int): m parameter of the matrix. Final matrix will be of size 2*m-1 

    Returns:
        List[List[float]]: resulting finite difference matrix

    """
    V = vander_matrix(m)
    D = normalization_matrix(m)
    V_inv = np.linalg.inv(V)
    D_inv = np.linalg.inv(D)
    A = D_inv @ V_inv
    return np.round(A, 5)


class cardinal_z_spline(object):
    """

    Cardinal Z-spline with arbitrary number of interpolation knots.
    It uses sympy functions for more accurate analysis of the properties of Z-splines.

    Example usage:
      >>> f = lambda x: math.sin(x)
      >>> m = 4
      >>> z_spline = cardinal_z_spline(m = m, f = f)
      >>> x = [...]
      >>> interpolation_result = z_spline.apply(x)

    """

    def __init__(self, m: int, f: Function) -> None:
        """

        Initilizes Z-spline.

        Parameters:
          m (int): m parameter of the spline.
          f (Function): function that will be approximated.

        Attributes:
          m (int): m parameter of the spline.
          A (List[List[float]]): Finite difference square matrix of size 2*m - 1.
          y_m (List[float]): y values of the knots.

        """
        self.m = m
        self.A = finite_difference_matrix(m)
        self.y_m = [f(y) for y in range(-m, m)]

    def __b1(self, nu: int) -> float:
        coef = 1.0
        for i in range(nu):
            coef *= (-self.m - i)

        return coef / factorial(nu)

    def __b0(self, nu: int) -> float:
        coef = 1.0
        for i in range(nu):
            coef *= (self.m + i)

        return coef / factorial(nu)

    def __B0(self, x: sym.Symbol, p: int, j: int) -> sym.Function:
        """

        Parameters:
          x (sym.Symbol): sympy symbol. For more info see https://docs.sympy.org/latest/explanation/glossary.html#term-Symbol
          p (int): Order of the derivative (zero is considered to be the function itself).
          j (int): Current knot.

        Returns:
          sym.Function: function for future evaluation

        """
        S = 0
        for nu in range(self.m - p):
            S += self.__b0(nu) * ((x - j)**nu)

        return (((x - j)**p) * S * ((j + 1 - x)**self.m)) / factorial(p)

    def __B1(self, x: sym.Symbol, p: int, j: int) -> sym.Function:
        """

        Parameters:
          x (sym.Symbol): sympy symbol. For more info see https://docs.sympy.org/latest/explanation/glossary.html#term-Symbol
          p (int): Order of the derivative (zero is considered to be the function itself).
          j (int): Current knot.

        Returns:
          sym.Function: function for future evaluation

        """
        S = 0
        for nu in range(self.m - p):
            S += self.__b1(nu)*((x - j - 1)**nu)

        return (((x - j - 1)**p) * S * ((x - j)**self.m)) / factorial(p)

    def __Z_p(self, p: int, j: int) -> float:
        """

        Calculates pth derivative of the spline at the given knot.

        Parameters:
          p (int): Order of the derivative (zero is considered to be the function itself).
          j (int): Current knot.

        Returns:
          float: pth derivative of the spline at jth knot.

        """
        sec_id = self.m-j-1
        # NOTE(kon3gor): second index might be outside of the boundaries for extreme knots.
        # Using zero in this case is an expected behavior.
        if sec_id >= 2*self.m - 1 or sec_id < 0:
            return 0

        return self.A[p][sec_id]

    def Z_m(self, shift: int = 0) -> Piecewise:
        """

        Calculates basis Z-functions of the Z-spline.

        Parameters:
          shift (int): Delta by which the center knot of the spline should be shifted. Defaults to 0.

        Returns:
          Piecewise: sympy Piecewise that describes the Z-spline between each pair of consecutive knots.

        """
        x = sym.Symbol('x')

        # NOTE(kon3gor): Z-spline is not defined outside of the boundaries (-m and m).
        pieces = [
            (0, (x - shift < -self.m) | (x - shift > self.m))
        ]

        # NOTE(kon3gor): For each knot.
        for j in range(-self.m, self.m):
            z = 0
            for p in range(self.m):
                z += self.__Z_p(p, j)*self.__B0(x - shift, p, j) + \
                    self.__Z_p(p, j + 1)*self.__B1(x - shift, p, j)

            # NOTE(kon3gor): Function z is defined only between current knot and the next knot.
            pieces.append(
                (z, (x - shift >= j) & (x - shift <= j + 1))
            )

        return Piecewise(*pieces)

    def apply(self, x: List[float]) -> List[float]:
        """

        Evaluates Z-spline at each given point. 
        Uses derivative method because 0th derivative is the function itself.

        Parameters:
          x (List[float]): points for the spline.

        Returns:
          List[float]: y values of the spline at given points.

        """
        return self.derivative(x, 0)

    def derivative(self, data: List[float], order: int) -> List[float]:
        """

        Evaluates an arbitrary derivative of the z-spline at each given point.

        Parameters:
          data (List[float]): points for the spline.
          order (int): order of the derivative

        Returns:
          List[float]: y values of the spline at given points.

        """
        x = sym.Symbol('x')

        Z = self.Z_m()
        for _ in range(order):
            Z = Z.diff(x)

        base = sym.lambdify(x, Z)
        y = []
        for x_i in data:
            y_i = 0
            for j in range(-self.m, self.m):
                y_i += self.y_m[self.m + j] * base(x_i - j)

            y.append(y_i)

        return y
