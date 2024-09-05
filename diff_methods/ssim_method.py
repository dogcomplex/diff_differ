import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from .base_method import BaseDiffMethod

class SSIMMethodBase(BaseDiffMethod):
    def __init__(self, beta, threshold):
        self.beta = beta
        self.threshold = threshold

    def generate_diff(self, img1, img2):
        ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
        ssim_diff = np.power(1 - ssim_map, self.beta)
        ssim_diff[ssim_diff < self.threshold] = 0
        ssim_diff = (ssim_diff * 255).clip(0, 255).astype(np.uint8)
        
        if ssim_diff.ndim == 2:
            ssim_diff = np.stack([ssim_diff] * 3, axis=-1)
        return ssim_diff

    def recreate_screenshot(self, earlier_screenshot, delta):
        ssim_map = delta.astype(np.float32) / 255.0
        alpha = np.power(ssim_map, 1/self.beta)
        recreated = ((1 - alpha) * earlier_screenshot).clip(0, 255).astype(np.uint8)
        return recreated

    @property
    def name(self):
        return f'ssim_b{self.beta}_t{self.threshold}'



class SSIMMethod1(SSIMMethodBase):
    def __init__(self):
        super().__init__(beta=4.0, threshold=0.2)

class SSIMMethod2(SSIMMethodBase):
    def __init__(self):
        super().__init__(beta=3.0, threshold=0.15)

class SSIMMethod3(SSIMMethodBase):
    def __init__(self):
        super().__init__(beta=5.0, threshold=0.25)
