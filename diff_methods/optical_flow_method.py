import cv2
import numpy as np
from .base_method import BaseDiffMethod

class OpticalFlowMethod(BaseDiffMethod):
    def __init__(self):
        self.flow = None

    def generate_diff(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        self.flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert flow to RGB for visualization
        hsv = np.zeros_like(img1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return rgb

    def recreate_screenshot(self, earlier_screenshot, delta):
        h, w = earlier_screenshot.shape[:2]
        flow = cv2.resize(self.flow, (w, h))
        
        # Use flow to warp the earlier screenshot
        flow_map = np.column_stack((flow[..., 0].flatten(), flow[..., 1].flatten())).reshape(h, w, 2)
        recreated = cv2.remap(earlier_screenshot, flow_map, None, cv2.INTER_LINEAR)
        
        return recreated

    @property
    def name(self):
        return 'optical_flow'

