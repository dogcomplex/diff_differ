import cv2
import numpy as np
from .base_method import BaseDiffMethod

class EnhancedGridDiffMethod(BaseDiffMethod):
    def __init__(self, grid_size=9, base_threshold=8, top_border=4, left_border=4, bottom_border=2, right_border=2, overlap=2, use_dynamic_threshold=True, use_adaptive_threshold=False):
        self.grid_size = grid_size
        self.base_threshold = base_threshold
        self.top_border = top_border
        self.left_border = left_border
        self.bottom_border = bottom_border
        self.right_border = right_border
        self.overlap = overlap
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_adaptive_threshold = use_adaptive_threshold

    def generate_diff(self, img1, img2):
        h, w = img1.shape[:2]
        effective_h = h - self.top_border - self.bottom_border
        effective_w = w - self.left_border - self.right_border
        cell_h, cell_w = effective_h // self.grid_size, effective_w // self.grid_size

        diff = np.zeros_like(img1, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.use_adaptive_threshold:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            adaptive_threshold = cv2.adaptiveThreshold(cv2.absdiff(img1_gray, img2_gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = max(0, self.top_border + i * cell_h - self.overlap)
                y2 = min(h, self.top_border + (i + 1) * cell_h + self.overlap)
                x1 = max(0, self.left_border + j * cell_w - self.overlap)
                x2 = min(w, self.left_border + (j + 1) * cell_w + self.overlap)

                cell1 = img1[y1:y2, x1:x2]
                cell2 = img2[y1:y2, x1:x2]

                cell_diff = np.mean(np.abs(cell2.astype(np.float32) - cell1.astype(np.float32)))

                if self.use_dynamic_threshold:
                    local_threshold = self.base_threshold * (1 + np.std(cell1) / 128)
                else:
                    local_threshold = self.base_threshold

                if self.use_adaptive_threshold:
                    cell_adaptive_threshold = adaptive_threshold[y1:y2, x1:x2]
                    if np.mean(cell_adaptive_threshold) > local_threshold:
                        diff[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
                        mask[y1:y2, x1:x2] = 255
                elif cell_diff > local_threshold:
                    diff[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
                    mask[y1:y2, x1:x2] = 255

        diff_with_alpha = np.dstack((diff, mask))
        return diff_with_alpha

    def recreate_screenshot(self, earlier_screenshot, delta):
        h, w = earlier_screenshot.shape[:2]
        effective_h = h - self.top_border - self.bottom_border
        effective_w = w - self.left_border - self.right_border
        cell_h, cell_w = effective_h // self.grid_size, effective_w // self.grid_size

        recreated = earlier_screenshot.copy()
        mask = delta[:,:,3]  # Alpha channel of the delta

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = max(0, self.top_border + i * cell_h - self.overlap)
                y2 = min(h, self.top_border + (i + 1) * cell_h + self.overlap)
                x1 = max(0, self.left_border + j * cell_w - self.overlap)
                x2 = min(w, self.left_border + (j + 1) * cell_w + self.overlap)

                if np.mean(mask[y1:y2, x1:x2]) > 127:
                    recreated[y1:y2, x1:x2] = delta[y1:y2, x1:x2, :3]

        return recreated

    def reverse_diff(self, delta):
        reversed_delta = np.zeros_like(delta)
        reversed_delta[:,:,3] = delta[:,:,3]  # Keep the same mask
        reversed_delta[:,:,:3] = 255 - delta[:,:,:3]  # Invert the color channels
        return reversed_delta

    def recreate_previous_screenshot(self, later_screenshot, delta):
        return self.recreate_screenshot(later_screenshot, self.reverse_diff(delta))

    @property
    def name(self):
        return f'enhanced_grid_diff_{self.grid_size}x{self.grid_size}_t{self.base_threshold}_tb{self.top_border}_lb{self.left_border}_bb{self.bottom_border}_rb{self.right_border}_o{self.overlap}_dt{int(self.use_dynamic_threshold)}_at{int(self.use_adaptive_threshold)}'

    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip',
            'analysis': 'skip',
            'tune': True
        }
