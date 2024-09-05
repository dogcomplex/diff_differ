import cv2
import numpy as np
from .base_method import BaseDiffMethod

class CombinedMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2):
        img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
        delta_y = cv2.subtract(img2[:,:,0].astype(np.int16), img1[:,:,0].astype(np.int16))
        delta_cb = cv2.subtract(img2_ycbcr[:,:,1].astype(np.int16), img1_ycbcr[:,:,1].astype(np.int16))
        delta_cr = cv2.subtract(img2_ycbcr[:,:,2].astype(np.int16), img1_ycbcr[:,:,2].astype(np.int16))
        return np.dstack((delta_y, delta_cb, delta_cr))

    def recreate_screenshot(self, earlier_screenshot, delta):
        earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
        recreated_y = np.clip(earlier_screenshot[:,:,0].astype(np.int16) + delta[:,:,0].astype(np.int16), 0, 255).astype(np.uint8)
        recreated_cb = np.clip(earlier_ycbcr[:,:,1].astype(np.int16) + delta[:,:,1].astype(np.int16), 0, 255).astype(np.uint8)
        recreated_cr = np.clip(earlier_ycbcr[:,:,2].astype(np.int16) + delta[:,:,2].astype(np.int16), 0, 255).astype(np.uint8)
        recreated_ycbcr = np.dstack((recreated_y, recreated_cb, recreated_cr))
        return cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)

    @property
    def name(self):
        return 'combined'

