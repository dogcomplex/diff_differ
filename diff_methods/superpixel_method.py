import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from .base_method import BaseDiffMethod
from config import SUPERPIXEL_SEGMENTS, SUPERPIXEL_COMPACTNESS, SUPERPIXEL_CHANGE_THRESHOLD

class SuperpixelMethod(BaseDiffMethod):
    def generate_superpixels(self, img, n_segments, compactness):
        return slic(img, n_segments=n_segments, compactness=compactness)

    def generate_diff(self, img1, img2):
        img1_float = img_as_float(img1)
        img2_float = img_as_float(img2)

        segments1 = self.generate_superpixels(img1_float, SUPERPIXEL_SEGMENTS, SUPERPIXEL_COMPACTNESS)
        segments2 = self.generate_superpixels(img2_float, SUPERPIXEL_SEGMENTS, SUPERPIXEL_COMPACTNESS)

        diff = np.zeros_like(img1, dtype=np.float32)
        
        for segment in np.unique(segments1):
            mask1 = segments1 == segment
            overlap = np.bincount(segments2[mask1].ravel())
            best_segment2 = overlap.argmax()
            mask2 = segments2 == best_segment2
            combined_mask = mask1 & mask2
            superpixel_diff = img2_float[combined_mask] - img1_float[combined_mask]
            mean_abs_diff = np.mean(np.abs(superpixel_diff), axis=0)
            
            if np.any(mean_abs_diff > SUPERPIXEL_CHANGE_THRESHOLD):
                diff[combined_mask] = superpixel_diff
        
        diff = np.clip(diff * 127.5 + 127.5, 0, 255).astype(np.uint8)
        return diff

    def recreate_screenshot(self, earlier_screenshot, delta):
        mask = np.any(np.abs(delta.astype(np.int16) - 127) > 1, axis=-1)
        recreated = earlier_screenshot.copy()
        recreated[mask] = np.clip(earlier_screenshot[mask].astype(np.float32) + (delta[mask].astype(np.float32) - 127.5) * 2, 0, 255).astype(np.uint8)
        return recreated

    @property
    def name(self):
        return 'superpixel'
