import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from .base_method import BaseDiffMethod
from config import SSIM_BETA, SSIM_THRESHOLD

class SSIMMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2):
        ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
        ssim_diff = np.power(1 - ssim_map, SSIM_BETA)
        ssim_diff[ssim_diff < SSIM_THRESHOLD] = 0
        ssim_diff = (ssim_diff * 255).clip(0, 255).astype(np.uint8)
        
        if ssim_diff.ndim == 2:
            ssim_diff = np.stack([ssim_diff] * 3, axis=-1)
        return ssim_diff

    def recreate_screenshot(self, earlier_screenshot, delta, next_screenshot):
        ssim_map = delta.astype(np.float32) / 255.0
        alpha = np.power(ssim_map, 0.25)
        recreated = ((1 - alpha) * earlier_screenshot + alpha * next_screenshot).clip(0, 255).astype(np.uint8)
        return recreated

    @property
    def name(self):
        return 'ssim'

    @property
    def config(self):
        return {
            'diff': 'overwrite',
            'recreation': 'overwrite'
        }
