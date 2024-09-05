import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, segmentation, morphology
from .base_method import BaseDiffMethod
import time

class MembraneMethod(BaseDiffMethod):
    def generate_diff(self, img1, img2, edge_threshold=0.1, region_threshold=0.05):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        edges1 = filters.sobel(gray1)
        edges2 = filters.sobel(gray2)
        edge_diff = np.abs(edges2 - edges1)
        
        mask = edge_diff > edge_threshold
        mask = morphology.dilation(mask, morphology.disk(3))
        
        distance = ndimage.distance_transform_edt(~mask)
        markers = filters.threshold_local(distance, 31, offset=0.1)
        markers = morphology.label(markers)
        
        segments = segmentation.watershed(-distance, markers, mask=mask)
        
        diff = np.zeros_like(img1, dtype=np.float32)
        for segment in np.unique(segments):
            if segment == 0:
                continue
            segment_mask = segments == segment
            segment_diff = img2[segment_mask].astype(np.float32) - img1[segment_mask].astype(np.float32)
            if np.mean(np.abs(segment_diff)) > region_threshold:
                diff[segment_mask] = segment_diff
        
        diff = np.clip(diff * 127.5 + 127.5, 0, 255).astype(np.uint8)
        return diff

    def recreate_screenshot(self, earlier_screenshot, delta, next_screenshot):
        mask = np.any(np.abs(delta.astype(np.int16) - 127) > 1, axis=-1)
        recreated = earlier_screenshot.copy()
        recreated[mask] = np.clip(earlier_screenshot[mask].astype(np.float32) + (delta[mask].astype(np.float32) - 127.5) * 2, 0, 255).astype(np.uint8)
        return recreated

    @property
    def name(self):
        return 'membrane'

    @property
    def config(self):
        return {
            'diff': 'overwrite',
            'recreation': 'overwrite'
        }
