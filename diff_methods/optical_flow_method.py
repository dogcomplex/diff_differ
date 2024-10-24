import cv2
import numpy as np
from .base_method import BaseDiffMethod

class OpticalFlowMethod(BaseDiffMethod):
    def __init__(self):
        self.flow = None

    def generate_diff(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions")

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        self.flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        if self.flow is None or self.flow.shape[:2] != img1.shape[:2]:
            raise ValueError("Failed to calculate optical flow or flow dimensions are incorrect")

        # Convert flow to RGB for visualization
        hsv = np.zeros_like(img1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return rgb

    def recreate_screenshot(self, earlier_screenshot, delta):
        if self.flow is None:
            raise ValueError("Optical flow has not been calculated. Call generate_diff first.")

        h, w = earlier_screenshot.shape[:2]
        if self.flow.shape[:2] != (h, w):
            self.flow = cv2.resize(self.flow, (w, h))

        # Use flow to warp the earlier screenshot
        flow_map = np.column_stack((self.flow[..., 0].flatten(), self.flow[..., 1].flatten())).reshape(h, w, 2)
        recreated = cv2.remap(earlier_screenshot, flow_map, None, cv2.INTER_LINEAR)
        
        return recreated

    def reverse_diff(self, delta):
        reversed_flow = -self.flow
        hsv = np.zeros_like(delta)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(reversed_flow[..., 0], reversed_flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @property
    def name(self):
        return 'optical_flow'

