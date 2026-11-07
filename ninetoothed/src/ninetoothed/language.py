import ast

try:
    from triton.language.extra import libdevice
except ImportError:
    # DTK 24.04 ships a Triton 2.1 fork whose extra namespace only exposes
    # ``cuda``.  Keep the public name available so kernels that do not use a
    # libdevice-only intrinsic can still be lowered on Hygon DCU.
    from triton.language.extra import cuda as libdevice

from ninetoothed.symbol import Symbol

__all__ = ["libdevice"]

LANGUAGE = "ninetoothed.language"


def call(func, *args, **kwargs):
    return Symbol(
        ast.Call(
            func=attribute(func).node,
            args=[Symbol(arg).node for arg in args],
            keywords=[
                ast.keyword(arg=kwarg, value=Symbol(kwargs[kwarg]).node)
                for kwarg in kwargs
            ],
        )
    )


def attribute(attr):
    return Symbol(ast.parse(f"{LANGUAGE}.{attr}", mode="eval").body)
