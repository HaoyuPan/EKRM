from PIL.Image import *

_fromarray = fromarray
from cupy import ndarray as _arr, asnumpy as _


def fromarray(obj: _arr, mode=None):
    return _fromarray(_(obj), mode)
