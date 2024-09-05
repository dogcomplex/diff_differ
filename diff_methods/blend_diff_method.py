import cv2
import numpy as np
from .base_method import BaseDiffMethod

class BlendDiffMethod(BaseDiffMethod):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def generate_diff(self, img1, img2):
        return cv2.addWeighted(img2, self.alpha, img1, 1 - self.alpha, 0)

    def recreate_screenshot(self, earlier_screenshot, delta):
        return cv2.addWeighted(delta, 1 / self.alpha, earlier_screenshot, 1 - (1 / self.alpha), 0)

    @property
    def name(self):
        return f'blend_diff_a{self.alpha}'
    
