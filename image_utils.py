import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import json
from datetime import datetime

def verify_recreation(screenshots_folder, recreated_folder, method_name):
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

    return {
        'avg_mse': float(avg_mse),
        'avg_ssim': avg_ssim,
        'perfect_matches': perfect_matches,
        'total_comparisons': valid_comparisons,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

def get_recent_cache(analyses_folder, method_name, max_age_seconds):
    cache_files = [f for f in os.listdir(analyses_folder) if f.startswith(f"verify_recreation_{method_name}_") and f.endswith('.json')]
    if not cache_files:
        return None
    
    cache_files.sort(reverse=True)
    latest_cache = cache_files[0]
    cache_path = os.path.join(analyses_folder, latest_cache)
    
    file_age = datetime.now().timestamp() - os.path.getmtime(cache_path)
    if file_age <= max_age_seconds:
        return cache_path
    
    return None

def analyze_high_mse_frames(screenshots_folder, recreated_folder, threshold=500, cached_result=None):
    print("===== ENTERING analyze_high_mse_frames FUNCTION =====")
    
    if cached_result:
        print("Using cached high MSE analysis results")
        return cached_result
    
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    recreated = sorted([f for f in os.listdir(recreated_folder) if f.endswith('.png')])
    
    high_mse_frames = []
    
    for i in range(1, len(screenshots)):
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        recreation = cv2.imread(os.path.join(recreated_folder, recreated[i-1]))
        
        if original is None or recreation is None or original.shape != recreation.shape:
            continue
        
        mse = float(np.mean((original.astype(np.float32) - recreation.astype(np.float32)) ** 2))
        
        if mse > threshold:
            high_mse_frames.append((i, mse))
    
    print(f"Found {len(high_mse_frames)} frames with MSE > {threshold}")
    detailed_analysis = []
    for frame, mse in high_mse_frames[:10]:
        print(f"Frame {frame}: MSE = {mse:.2f}")
        
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[frame]))
        recreation = cv2.imread(os.path.join(recreated_folder, recreated[frame-1]))
        diff = cv2.absdiff(original, recreation)
        
        changed_pixels = float(np.sum(diff > 0) / (diff.shape[0] * diff.shape[1] * diff.shape[2]) * 100)
        print(f"  Changed pixels: {changed_pixels:.2f}%")
        
        channel_changes = []
        for c, color in enumerate(['Blue', 'Green', 'Red']):
            channel_diff = diff[:,:,c]
            channel_changed = float(np.sum(channel_diff > 0) / (channel_diff.shape[0] * channel_diff.shape[1]) * 100)
            print(f"  {color} channel changes: {channel_changed:.2f}%")
            channel_changes.append((color, channel_changed))
        
        detailed_analysis.append({
            'frame': frame,
            'mse': mse,
            'changed_pixels': changed_pixels,
            'channel_changes': channel_changes
        })
    
    print("===== EXITING analyze_high_mse_frames FUNCTION =====")

    results = {
        'high_mse_count': len(high_mse_frames),
        'threshold': threshold,
        'detailed_analysis': detailed_analysis
    }
    return results

def verify_reversed_recreation(screenshots_folder, recreated_reversed_folder, method_name):
    print("===== ENTERING verify_reversed_recreation FUNCTION =====")
    
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    recreated_reversed = sorted([f for f in os.listdir(recreated_reversed_folder) if f.endswith('.png')])
    
    if len(screenshots) != len(recreated_reversed) + 1:
        print(f"Mismatch in number of screenshots: Original {len(screenshots)}, Recreated Reversed {len(recreated_reversed)}")
        return {
            'method': method_name,
            'avg_mse': 0,
            'avg_ssim': 0,
            'perfect_matches': 0,
            'total_comparisons': 0,
            'error': 'Mismatch in number of screenshots'
        }
    
    total_mse = 0
    total_ssim = 0
    perfect_matches = 0
    valid_comparisons = 0
    
    for i in range(len(recreated_reversed)):
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        reversed_rec = cv2.imread(os.path.join(recreated_reversed_folder, recreated_reversed[i]))
        
        if original is None or reversed_rec is None:
            print(f"Error reading images for comparison {i}")
            continue
        
        mse = np.mean((original - reversed_rec) ** 2)
        
        # Determine the appropriate window size for SSIM
        min_dim = min(original.shape[0], original.shape[1])
        win_size = min(7, min_dim - (min_dim % 2) + 1)  # Ensure win_size is odd and not larger than the image
        
        ssim_value = ssim(original, reversed_rec, win_size=win_size, channel_axis=2)
        
        total_mse += mse
        total_ssim += ssim_value
        valid_comparisons += 1
        
        if mse == 0:
            perfect_matches += 1
        
        if i % 100 == 0:
            print(f"Comparison {i}: MSE = {mse:.2f}, SSIM = {ssim_value:.4f}")
    
    avg_mse = total_mse / valid_comparisons if valid_comparisons > 0 else 0
    avg_ssim = total_ssim / valid_comparisons if valid_comparisons > 0 else 0
    
    print(f"Method: {method_name}")
    print(f"Average MSE: {avg_mse:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Perfect Matches: {perfect_matches}/{valid_comparisons}")
    
    print("===== EXITING verify_reversed_recreation FUNCTION =====")
    
    return {
        'method': method_name,
        'avg_mse': float(avg_mse),
        'avg_ssim': avg_ssim,
        'perfect_matches': perfect_matches,
        'total_comparisons': valid_comparisons
    }
