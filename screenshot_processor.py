import os
import cv2
import numpy as np
from config import *
from diff_generators import *
from image_utils import verify_recreation, analyze_high_mse_frames
from skimage.metrics import structural_similarity as ssim

def generate_diffs(screenshots_folder, diffs_folder):
    print("===== ENTERING generate_diffs FUNCTION =====")
    # ... (existing code)

def recreate_screenshots(screenshots_folder, delta_folder, recreated_folder, method):
    print(f"===== ENTERING recreate_screenshots ({method}) FUNCTION =====")
    # ... (existing code)

def process_screenshots():
    print("===== ENTERING process_screenshots FUNCTION =====")
    try:
        generate_diffs(SCREENSHOTS_FOLDER, DIFFS_FOLDER)
        
        for method in METHODS:
            print(f"\nTesting {method} method:")
            delta_folder = os.path.join(DIFFS_FOLDER, f"delta_{method if method != 'current' else 'diffs'}")
            recreated_folder = os.path.join(RECREATIONS_FOLDER, method)
            recreate_screenshots(SCREENSHOTS_FOLDER, delta_folder, recreated_folder, method)
            verify_recreation(SCREENSHOTS_FOLDER, recreated_folder)
            
            print(f"\nAnalyzing high MSE frames for {method} method:")
            analyze_high_mse_frames(SCREENSHOTS_FOLDER, recreated_folder)
    except Exception as e:
        print(f"===== ERROR IN process_screenshots: {str(e)} =====")
    print("===== EXITING process_screenshots FUNCTION =====")

def generate_ssim_diff(img1, img2, beta=SSIM_BETA, threshold=SSIM_THRESHOLD):
    ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
    ssim_diff = np.power(1 - ssim_map, beta)
    ssim_diff[ssim_diff < threshold] = 0
    ssim_diff = (ssim_diff * 255).clip(0, 255).astype(np.uint8)
    
    if ssim_diff.ndim == 2:
        ssim_diff = np.stack([ssim_diff] * 3, axis=-1)
    
    return ssim_diff