import sympy as sym
from sympy.abc import x
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lib.helpers import generate_data
from lib.zspline import cardinal_z_spline


def f() -> sym.Function:
    """
    Function that will be used fot Z-spline showcase.
    Change it if you want somthing different.
    """
    return sym.sin(x)  # NOTE(kon3gor): x is imported from sympy.abc for convenience


def showcase():
    m = int(input("Choose m value for the Z-spline: "))
    X = generate_data(-2*m, 2*m, 0.001)
    F = sym.lambdify(x, f())
    Y = [F(x_i) for x_i in X]

    spline = cardinal_z_spline(m=m, f=F)
    y_test = spline.apply(X)

    plt.plot(X, y_test, color="red")
    plt.plot(X, Y, color="green")
    plt.savefig("showcase.png")

    print("Saved result into showcase.png")


def derivatives_table():
    m = 4

    func = f()
    spline = cardinal_z_spline(m=m, f=sym.lambdify(x, func))

    expected = pd.Series()
    actual = pd.Series()

    for n in range(1, 2*m):
        func = func.diff(x)
        F = sym.lambdify(x, func)

        expected[n] = F(0)
        actual[n] = spline.derivative([0], n)[0]

    df = pd.DataFrame({"expected": expected, "actual": actual})
    df["diff"] = df["expected"] - df["actual"]
    pd.options.display.float_format = '{:.6f}'.format
    print(df)


def sync_comparison():
    max_m = 4
    X = generate_data(-4, 4, 0.001)
    plt.plot(X, [np.sinc(x) for x in X], color="g")

    for m in range(1, max_m + 1):
        base = cardinal_z_spline(m=m, f=lambda x: x).Z_m()
        basis_f = sym.lambdify(sym.Symbol('x'), base)

        y_test = [basis_f(x) for x in X]

        plt.plot(X, y_test, color="r")

    plt.savefig("sync_comparison.png")
    print("saved into sync_comparison.png")


MODES = ["interpolation", "derivatives", "sync"]


def main(mode: str):
    if mode == MODES[0] or mode == "":
        showcase()
    elif mode == MODES[1]:
        derivatives_table()
    elif mode == MODES[2]:
        sync_comparison()
    else:
        print("No such mode selected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Z-spline showcase"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=MODES,
        help="Mode in which the program whould work. Interpolation will graph interpolation result for the given m value. Derivatives will print derivatives comparison. And sync will graph Z-splines along with ideal sync funcion."
    )

    args = parser.parse_args()

    main(args.mode)
