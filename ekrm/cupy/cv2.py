from cupy import asnumpy as _, ndarray as _Arr, array as _arr
from cv2 import findContours as npFindContours, fillPoly as npFillPoly, RETR_TREE, CHAIN_APPROX_SIMPLE

__keep = (RETR_TREE, CHAIN_APPROX_SIMPLE)


def findContours(image: _Arr, mode, method, *args, **kwargs):
    contours, hierarchy = npFindContours(_(image), mode, method, *args, **kwargs)
    return [_arr(i) for i in contours], _arr(hierarchy)


def fillPoly(img: _Arr, pts: list[_Arr], color, *args, **kwargs):
    return _arr(npFillPoly(_(img), [_(i) for i in pts], color, *args, **kwargs))
