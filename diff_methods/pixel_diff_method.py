import cv2
import numpy as np
from .base_method import BaseDiffMethod

class PixelDiffMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2):
        print("Generating pixel-wise diff...")
        delta = cv2.subtract(img2.astype(np.int16), img1.astype(np.int16))
        print("Pixel-wise diff generated.")
        return delta

    def recreate_screenshot(self, earlier_screenshot, delta):
        print("Recreating screenshot using pixel-wise diff method...")
        recreated = np.clip(earlier_screenshot.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
        print("Screenshot recreated.")
        return recreated

    @property
    def name(self):
        return 'pixel_diff'

    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip'
        }