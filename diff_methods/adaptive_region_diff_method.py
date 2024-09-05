import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.future import graph
from .base_method import BaseDiffMethod

class AdaptiveRegionDiffMethod(BaseDiffMethod):
    def __init__(self, n_segments=100, compactness=10, sigma=1):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma

    def generate_diff(self, img1, img2):
        # Generate superpixels for both images
        segments1 = slic(img1, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma)
        segments2 = slic(img2, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma)

        # Initialize diff and mask
        diff = np.zeros_like(img1)
        mask = np.zeros(img1.shape[:2], dtype=np.uint8)

        # Compare corresponding superpixels
        for segment_id in np.unique(segments1):
            mask1 = segments1 == segment_id
            mask2 = segments2 == segment_id

            region1 = img1[mask1]
            region2 = img2[mask2]

            if not np.array_equal(region1, region2):
                diff[mask1] = img2[mask1]
                mask[mask1] = 255

        # Combine diff and mask into a 4-channel image (BGRA)
        diff_with_alpha = np.dstack((diff, mask))
        return diff_with_alpha

    def recreate_screenshot(self, earlier_screenshot, delta):
        mask = delta[:,:,3]
        recreated = earlier_screenshot.copy()
        recreated[mask > 0] = delta[mask > 0, :3]
        return recreated

    @property
    def name(self):
        return f'adaptive_region_diff_n{self.n_segments}_c{self.compactness}_s{self.sigma}'
