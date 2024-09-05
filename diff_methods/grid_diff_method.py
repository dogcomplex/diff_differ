import cv2
import numpy as np
from .base_method import BaseDiffMethod

class GridDiffMethod(BaseDiffMethod):
    def __init__(self, grid_size=9, threshold=10):
        self.grid_size = grid_size
        self.threshold = threshold

    def generate_diff(self, img1, img2):
        h, w = img1.shape[:2]
        cell_h, cell_w = h // self.grid_size, w // self.grid_size

        diff = np.zeros_like(img1)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w

                cell1 = img1[y1:y2, x1:x2]
                cell2 = img2[y1:y2, x1:x2]

                cell_diff = np.mean(np.abs(cell2.astype(np.float32) - cell1.astype(np.float32)))

                if cell_diff > self.threshold:
                    diff[y1:y2, x1:x2] = 255

        return diff

    def recreate_screenshot(self, earlier_screenshot, delta):
        h, w = earlier_screenshot.shape[:2]
        cell_h, cell_w = h // self.grid_size, w // self.grid_size

        recreated = earlier_screenshot.copy()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w

                if np.mean(delta[y1:y2, x1:x2]) > 127:
                    recreated[y1:y2, x1:x2] = 255 - earlier_screenshot[y1:y2, x1:x2]

        return recreated

    @property
    def name(self):
        return f'grid_diff_{self.grid_size}x{self.grid_size}_t{self.threshold}'

    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip'
        }
