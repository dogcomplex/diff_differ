from .current_method import CurrentMethod
from .ycbcr_method import YCbCrMethod
from .combined_method import CombinedMethod
from .ssim_method import SSIMMethod
from .superpixel_method import SuperpixelMethod
from .membrane_method import MembraneMethod

DIFF_METHODS = [
    CurrentMethod(),
    YCbCrMethod(),
    CombinedMethod(),
    SSIMMethod(),
    SuperpixelMethod(),
    MembraneMethod()
]

METHOD_DICT = {method.name: method for method in DIFF_METHODS}
