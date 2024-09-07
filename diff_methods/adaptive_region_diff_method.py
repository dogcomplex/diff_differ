import cv2
import numpy as np
from skimage.segmentation import slic
from .base_method import BaseDiffMethod
import time
import os

class AdaptiveRegionDiffMethod(BaseDiffMethod):
    def __init__(self, n_segments=100, compactness=20, sigma=1, min_area=10, threshold=0.15, max_regions=200, cumulative_threshold=10.0, global_threshold=0.1):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.min_area = min_area  # Reduced from 800 to 400 (20x20 pixels)
        self.threshold = threshold
        self.max_regions = max_regions  # Increased from 10 to 15 to allow for more detected regions
        self.cumulative_threshold = cumulative_threshold
        self.global_threshold = global_threshold

    def generate_diff(self, img1, img2):
        print(f"DEBUG: Parameters - min_area: {self.min_area}, threshold: {self.threshold}, max_regions: {self.max_regions}")
        print("DEBUG: Entering generate_diff method")

        # Create debug directory
        debug_dir = os.path.join('screenshots', 'diffs', 'debug', self.name)
        os.makedirs(debug_dir, exist_ok=True)

        timestamp = int(time.time())

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        pixel_diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        max_pixel_diff = np.max(pixel_diff)
        print(f"Max pixel diff: {max_pixel_diff:.2f}")

        if max_pixel_diff == 0:
            print("Images identical")
            return np.zeros_like(img1)

        # Global difference check
        global_diff = np.mean(pixel_diff)
        print(f"Global diff: {global_diff:.4f}")

        # Convert to grayscale for processing
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        cv2.imwrite(os.path.join(debug_dir, f'debug_diff_{timestamp}.png'), diff)
        print("DEBUG: Saved initial diff image")

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(os.path.join(debug_dir, f'debug_thresh_{timestamp}.png'), thresh)
        print("DEBUG: Saved thresholded image")

        # Apply morphological operations to remove noise and connect regions
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(debug_dir, f'debug_morph_{timestamp}.png'), thresh)
        print("DEBUG: Saved morphological operations result")

        # Find contours of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"DEBUG: Found {len(contours)} contours")

        # Filter contours based on area
        mask = np.zeros(img1.shape[:2], dtype=np.uint8)
        contours_kept = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                cv2.drawContours(mask, [contour], 0, 255, -1)
                contours_kept += 1
        print(f"DEBUG: Kept {contours_kept} contours out of {len(contours)}")
        cv2.imwrite(os.path.join(debug_dir, f'debug_mask_{timestamp}.png'), mask)
        print("DEBUG: Saved final mask")

        # Apply the mask to the original difference
        diff_color = cv2.absdiff(img1, img2)
        diff_masked = cv2.bitwise_and(diff_color, diff_color, mask=mask)
        cv2.imwrite(os.path.join(debug_dir, f'debug_diff_masked_{timestamp}.png'), diff_masked)
        print("DEBUG: Saved masked diff")

        # Create the final output with alpha channel
        diff_with_alpha = np.dstack((diff_masked, mask))

        if np.all(diff_with_alpha == 0):
            print("WARNING: Diff is all zeros!")

        print("DEBUG: Exiting generate_diff method")
        np.save(os.path.join(debug_dir, f'raw_diff_{timestamp}.npy'), diff_with_alpha)
        cv2.imwrite(os.path.join(debug_dir, f'final_diff_{timestamp}.png'), diff_with_alpha)
        print(f"Diff stats: min={np.min(diff_masked)}, max={np.max(diff_masked)}, mean={np.mean(diff_masked)}")
        return diff_with_alpha

    def recreate_screenshot(self, earlier_screenshot, delta):
        if earlier_screenshot.shape[:2] != delta.shape[:2]:
            raise ValueError(f"Shape mismatch: earlier_screenshot {earlier_screenshot.shape[:2]} vs delta {delta.shape[:2]}")
        
        mask = delta[:,:,3]
        recreated = earlier_screenshot.copy()
        
        # Only apply changes where the mask is non-zero
        changed_pixels = mask > 0
        recreated[changed_pixels] = earlier_screenshot[changed_pixels] + delta[changed_pixels, :3]
        
        # Clip values to ensure they're in the valid range for uint8
        recreated = np.clip(recreated, 0, 255).astype(np.uint8)
        
        return recreated

    @property
    def name(self):
        return f'adaptive_region_diff_n{self.n_segments}_c{self.compactness}_s{self.sigma}_a{self.min_area}_t{self.threshold}_m{self.max_regions}_ct{self.cumulative_threshold}_gt{self.global_threshold}'
