import cv2
import numpy as np
from .base_method import BaseDiffMethod

class PixelDiffMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2):
        delta = cv2.subtract(img2.astype(np.int16), img1.astype(np.int16))
        return delta

    def recreate_screenshot(self, earlier_screenshot, delta):
        recreated = np.clip(earlier_screenshot.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        return recreated

    def recreate_previous_screenshot(self, later_screenshot, delta):
        return np.clip(later_screenshot.astype(np.int16) - delta, 0, 255).astype(np.uint8)

    @property
    def name(self):
        return 'pixel_diff'
    
    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip',
            'analysis': 'skip',
            'tune': False
        }

