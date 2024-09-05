import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def verify_recreation(screenshots_folder, recreated_folder):
    print("===== ENTERING verify_recreation FUNCTION =====")
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    recreated = sorted([f for f in os.listdir(recreated_folder) if f.endswith('.png')])
    
    if len(screenshots) != len(recreated) + 1:
        print(f"Mismatch in number of screenshots: Original {len(screenshots)}, Recreated {len(recreated)}")
        return
    
    total_mse = 0
    total_ssim = 0
    perfect_matches = 0
    valid_comparisons = 0
    
    for i in range(1, len(screenshots)):
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        recreation = cv2.imread(os.path.join(recreated_folder, recreated[i-1]))
        
        if original is None or recreation is None or original.shape != recreation.shape:
            print(f"Error: Unable to compare screenshot {i}")
            continue
        
        mse = np.mean((original.astype(np.float32) - recreation.astype(np.float32)) ** 2)
        ssim_score = ssim(original, recreation, win_size=3, channel_axis=2)
        
        total_mse += mse
        total_ssim += ssim_score
        valid_comparisons += 1
        
        if np.array_equal(original, recreation):
            perfect_matches += 1
        
        if i % 100 == 0:
            print(f"Screenshot {i}: MSE = {mse:.2f}, SSIM = {ssim_score:.4f}")
    
    if valid_comparisons > 0:
        avg_mse = total_mse / valid_comparisons
        avg_ssim = total_ssim / valid_comparisons
        print(f"Average MSE: {avg_mse:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Perfect matches: {perfect_matches}/{valid_comparisons} ({perfect_matches/valid_comparisons*100:.2f}%)")
    else:
        print("No valid comparisons were made.")
    
    print("===== EXITING verify_recreation FUNCTION =====")

def analyze_high_mse_frames(screenshots_folder, recreated_folder, threshold=500):
    print("===== ENTERING analyze_high_mse_frames FUNCTION =====")
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    recreated = sorted([f for f in os.listdir(recreated_folder) if f.endswith('.png')])
    
    high_mse_frames = []
    
    for i in range(1, len(screenshots)):
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        recreation = cv2.imread(os.path.join(recreated_folder, recreated[i-1]))
        
        if original is None or recreation is None or original.shape != recreation.shape:
            continue
        
        mse = np.mean((original.astype(np.float32) - recreation.astype(np.float32)) ** 2)
        
        if mse > threshold:
            high_mse_frames.append((i, mse))
    
    print(f"Found {len(high_mse_frames)} frames with MSE > {threshold}")
    for frame, mse in high_mse_frames[:10]:
        print(f"Frame {frame}: MSE = {mse:.2f}")
        
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[frame]))
        recreation = cv2.imread(os.path.join(recreated_folder, recreated[frame-1]))
        diff = cv2.absdiff(original, recreation)
        
        changed_pixels = np.sum(diff > 0) / (diff.shape[0] * diff.shape[1] * diff.shape[2]) * 100
        print(f"  Changed pixels: {changed_pixels:.2f}%")
        
        for c, color in enumerate(['Blue', 'Green', 'Red']):
            channel_diff = diff[:,:,c]
            channel_changed = np.sum(channel_diff > 0) / (channel_diff.shape[0] * channel_diff.shape[1]) * 100
            print(f"  {color} channel changes: {channel_changed:.2f}%")
    
    print("===== EXITING analyze_high_mse_frames FUNCTION =====")
