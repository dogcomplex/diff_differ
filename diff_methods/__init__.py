from .pixel_diff_method import PixelDiffMethod
from .ycbcr_method import YCbCrMethod
from .combined_method import CombinedMethod
from .ssim_method import SSIMMethod1, SSIMMethod2, SSIMMethod3
from .superpixel_method import SuperpixelMethod
from .membrane_method import MembraneMethod
from .optical_flow_method import OpticalFlowMethod
from .blend_diff_method import BlendDiffMethod
from .feature_based_method import FeatureBasedMethod
from .grid_diff_method import GridDiffMethod

DIFF_METHODS = [
    PixelDiffMethod(),
    YCbCrMethod(),
    CombinedMethod(),
    SSIMMethod1(),
    SSIMMethod2(),
    SSIMMethod3(),
    SuperpixelMethod(),
    MembraneMethod(),
    OpticalFlowMethod(),
    BlendDiffMethod(alpha=0.5),
    FeatureBasedMethod(),
    GridDiffMethod(grid_size=9, threshold=10, top_border=1, left_border=1),
    GridDiffMethod(grid_size=9, threshold=10, top_border=1, left_border=1, bottom_border=1, right_border=1)
]

METHOD_DICT = {method.name: method for method in DIFF_METHODS}
