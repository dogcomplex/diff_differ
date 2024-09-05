import cv2
import numpy as np
from .base_method import BaseDiffMethod

class YCbCrMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2):
        img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
        return cv2.subtract(img2_ycbcr.astype(np.int16), img1_ycbcr.astype(np.int16))

    def recreate_screenshot(self, earlier_screenshot, delta, next_screenshot):
        earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
        recreated_ycbcr = np.clip(earlier_ycbcr.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
        return cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)

    def recreate_screenshot(self, earlier_screenshot, delta, next_screenshot):
        earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
        recreated_ycbcr = np.clip(earlier_ycbcr.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
        return cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)

    @property
    def name(self):
        return 'ycbcr'

    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip'
        }
