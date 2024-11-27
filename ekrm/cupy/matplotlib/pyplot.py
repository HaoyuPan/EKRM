import cupy
from matplotlib.container import BarContainer
from matplotlib.patches import Patch
from matplotlib.pyplot import *
from matplotlib.pyplot import barh as _barh, bar as _bar, hist as _hist, imshow as _imshow, pie as _pie

__keep = (barh, Patch)
_ = cupy.asnumpy
_Arr = cupy.ndarray


def bar(x: _Arr, height: _Arr, *args, **kwargs) -> BarContainer:
    return _bar(_(x), _(height), *args, **kwargs)


def barh(y: _Arr, width: _Arr, *args, **kwargs) -> BarContainer:
    return _barh(_(y), _(width), *args, **kwargs)


def hist(x: _Arr, *args, **kwargs):
    return _hist(_(x), *args, **kwargs)


def imshow(X: _Arr, *args, **kwargs):
    return _imshow(_(X), *args, **kwargs)


def pie(x: _Arr, *args, **kwargs):
    return _pie(_(x), *args, **kwargs)
