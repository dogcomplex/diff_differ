import cv2
import numpy as np
from .base_method import BaseDiffMethod

class CurrentMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2):
        return cv2.subtract(img2.astype(np.int16), img1.astype(np.int16))

    def recreate_screenshot(self, earlier_screenshot, delta, next_screenshot):
        return np.clip(earlier_screenshot.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)

    @property
    def name(self):
        return 'current'

    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip'
        }
