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
from .adaptive_region_diff_method import AdaptiveRegionDiffMethod
from .enhanced_grid_diff_method import EnhancedGridDiffMethod

# commented-out lines are DISABLED.  Please do not remove or change these from code
# this is just the easiest way to flip them on/off
DIFF_METHODS = [
    PixelDiffMethod(),
    #YCbCrMethod(),
    #CombinedMethod(),
    #SSIMMethod1(),
    #SSIMMethod2(),
    #SSIMMethod3(),
    #SuperpixelMethod(),
    #MembraneMethod(),
    #OpticalFlowMethod(),
    #BlendDiffMethod(alpha=0.5),
    #FeatureBasedMethod(),
    #GridDiffMethod(grid_size=9, threshold=10, top_border=0, left_border=0),
    #GridDiffMethod(grid_size=9, threshold=10, top_border=3, left_border=3, bottom_border=2, right_border=2),
    #GridDiffMethod(grid_size=9, threshold=10, top_border=4, left_border=4, bottom_border=2, right_border=2),
    #GridDiffMethod(grid_size=9, threshold=10, top_border=2, left_border=3, bottom_border=2, right_border=2),
    #GridDiffMethod(grid_size=16, threshold=10, top_border=4, left_border=4, bottom_border=2, right_border=2),
    #GridDiffMethod(grid_size=16, threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2),
    #GridDiffMethod(grid_size=9, threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2),
    GridDiffMethod(grid_size=9, threshold=12, top_border=4, left_border=4, bottom_border=2, right_border=2),
    GridDiffMethod(grid_size=9, threshold=10, top_border=5, left_border=5, bottom_border=3, right_border=3),
    #GridDiffMethod(grid_size=12, threshold=10, top_border=4, left_border=4, bottom_border=2, right_border=2),
    EnhancedGridDiffMethod(grid_size=9, base_threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2, overlap=2, use_dynamic_threshold=True, use_adaptive_threshold=False),
    EnhancedGridDiffMethod(grid_size=9, base_threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2, overlap=2, use_dynamic_threshold=True, use_adaptive_threshold=True),
    EnhancedGridDiffMethod(grid_size=9, base_threshold=6, top_border=4, left_border=4, bottom_border=2, right_border=2, overlap=2, use_dynamic_threshold=True, use_adaptive_threshold=False),
    #EnhancedGridDiffMethod(grid_size=16, base_threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2, overlap=2, use_dynamic_threshold=True, use_adaptive_threshold=False),
    #EnhancedGridDiffMethod(grid_size=18, base_threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2, overlap=2, use_dynamic_threshold=True, use_adaptive_threshold=False),
    AdaptiveRegionDiffMethod(),
]

METHOD_DICT = {method.name: method for method in DIFF_METHODS}
