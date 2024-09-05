import cv2
import numpy as np
from .base_method import BaseDiffMethod

class GridDiffMethod(BaseDiffMethod):
    def __init__(self, grid_size=9, threshold=10, border=0):
        self.grid_size = grid_size
        self.threshold = threshold
        self.border = border

    def generate_diff(self, img1, img2):
        h, w = img1.shape[:2]
        effective_h, effective_w = h - 2*self.border, w - 2*self.border
        cell_h, cell_w = effective_h // self.grid_size, effective_w // self.grid_size

        diff = np.zeros_like(img1, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = self.border + i * cell_h
                y2 = self.border + (i + 1) * cell_h
                x1 = self.border + j * cell_w
                x2 = self.border + (j + 1) * cell_w

                cell1 = img1[y1:y2, x1:x2]
                cell2 = img2[y1:y2, x1:x2]

                cell_diff = np.mean(np.abs(cell2.astype(np.float32) - cell1.astype(np.float32)))

                if cell_diff > self.threshold:
                    diff[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
                    mask[y1:y2, x1:x2] = 255

        # Combine diff and mask into a 4-channel image (BGRA)
        diff_with_alpha = np.dstack((diff, mask))
        return diff_with_alpha

    def recreate_screenshot(self, earlier_screenshot, delta):
        h, w = earlier_screenshot.shape[:2]
        effective_h, effective_w = h - 2*self.border, w - 2*self.border
        cell_h, cell_w = effective_h // self.grid_size, effective_w // self.grid_size

        recreated = earlier_screenshot.copy()
        mask = delta[:,:,3]  # Alpha channel of the delta

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = self.border + i * cell_h
                y2 = self.border + (i + 1) * cell_h
                x1 = self.border + j * cell_w
                x2 = self.border + (j + 1) * cell_w

                if np.mean(mask[y1:y2, x1:x2]) > 127:
                    recreated[y1:y2, x1:x2] = delta[y1:y2, x1:x2, :3]

        return recreated

    @property
    def name(self):
        return f'grid_diff_{self.grid_size}x{self.grid_size}_t{self.threshold}_b{self.border}'

