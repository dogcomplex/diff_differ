print("===== SCRIPT STARTED =====")

try:
    import cv2
    import os
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    print("===== IMPORTS SUCCESSFUL =====")
except ImportError as e:
    print(f"===== IMPORT ERROR: {e} =====")

def generate_diffs(screenshots_folder, diffs_folder):
    print("===== ENTERING generate_diffs FUNCTION =====")
    delta_diffs_folder = os.path.join(diffs_folder, "delta_diffs")
    delta_ycbcr_folder = os.path.join(diffs_folder, "delta_ycbcr")
    delta_combined_folder = os.path.join(diffs_folder, "delta_combined")
    delta_ssim_folder = os.path.join(diffs_folder, "delta_ssim")
    
    os.makedirs(delta_diffs_folder, exist_ok=True)
    os.makedirs(delta_ycbcr_folder, exist_ok=True)
    os.makedirs(delta_combined_folder, exist_ok=True)
    os.makedirs(delta_ssim_folder, exist_ok=True)
    
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])

    for i in range(len(screenshots) - 1):
        img1 = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        img2 = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
        
        # Method 1: Current method (int16)
        delta = cv2.subtract(img2.astype(np.int16), img1.astype(np.int16))
        cv2.imwrite(os.path.join(delta_diffs_folder, f"diff_delta_{i:04d}_{i+1:04d}.png"), delta)
        
        # Method 2: YCbCr color space
        img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
        delta_ycbcr = cv2.subtract(img2_ycbcr.astype(np.int16), img1_ycbcr.astype(np.int16))
        cv2.imwrite(os.path.join(delta_ycbcr_folder, f"diff_ycbcr_{i:04d}_{i+1:04d}.png"), delta_ycbcr)
        
        # Method 3: Combined method (YCbCr for chroma, RGB for luma)
        delta_y = cv2.subtract(img2[:,:,0].astype(np.int16), img1[:,:,0].astype(np.int16))
        delta_cb = cv2.subtract(img2_ycbcr[:,:,1].astype(np.int16), img1_ycbcr[:,:,1].astype(np.int16))
        delta_cr = cv2.subtract(img2_ycbcr[:,:,2].astype(np.int16), img1_ycbcr[:,:,2].astype(np.int16))
        delta_combined = np.dstack((delta_y, delta_cb, delta_cr))
        cv2.imwrite(os.path.join(delta_combined_folder, f"diff_combined_{i:04d}_{i+1:04d}.png"), delta_combined)
        
        # Method 4: SSIM-based diff
        ssim_diff = generate_ssim_diff(img1, img2)
        cv2.imwrite(os.path.join(delta_ssim_folder, f"diff_ssim_{i:04d}_{i+1:04d}.png"), ssim_diff)

    print(f"Generated diff images for {len(screenshots) - 1} pairs of screenshots.")
    print("===== EXITING generate_diffs FUNCTION =====")

def generate_ssim_diff(img1, img2):
    ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
    ssim_diff = ((1 - ssim_map) * 255).astype(np.uint8)
    return cv2.cvtColor(ssim_diff, cv2.COLOR_GRAY2BGR)

def generate_ssim_diff(img1, img2, beta=4.0, threshold=0.2):
    ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
    
    # Apply a non-linear transformation to make the diff more aggressive
    ssim_diff = np.power(1 - ssim_map, beta)
    
    # Apply a threshold to further emphasize changes
    ssim_diff[ssim_diff < threshold] = 0
    
    return (ssim_diff * 255).astype(np.uint8)

def generate_ssim_diff(img1, img2, beta=2.0, threshold=0.1):
    ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
    
    # Apply a non-linear transformation to make the diff more aggressive
    ssim_diff = np.power(1 - ssim_map, beta)
    
    # Apply a threshold to further emphasize changes
    ssim_diff[ssim_diff < threshold] = 0
    
    return (ssim_diff * 255).astype(np.uint8)

def recreate_screenshots(screenshots_folder, delta_folder, recreated_folder, method):
    print(f"===== ENTERING recreate_screenshots ({method}) FUNCTION =====")
    os.makedirs(recreated_folder, exist_ok=True)
    
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    delta_files = sorted([f for f in os.listdir(delta_folder) if f.endswith('.png')])
    
    perfect_recreations = 0
    total_recreations = len(delta_files)
    
    for i in range(len(delta_files)):
        earlier_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        next_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
        
        if method == 'current':
            delta = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_UNCHANGED).astype(np.int16)
            recreated = np.clip(earlier_screenshot.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        elif method == 'ycbcr':
            delta = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_UNCHANGED).astype(np.int16)
            earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
            recreated_ycbcr = np.clip(earlier_ycbcr.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            recreated = cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)
        elif method == 'combined':
            delta = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_UNCHANGED).astype(np.int16)
            recreated_y = np.clip(earlier_screenshot[:,:,0].astype(np.int16) + delta[:,:,0], 0, 255).astype(np.uint8)
            earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
            recreated_cb = np.clip(earlier_ycbcr[:,:,1].astype(np.int16) + delta[:,:,1], 0, 255).astype(np.uint8)
            recreated_cr = np.clip(earlier_ycbcr[:,:,2].astype(np.int16) + delta[:,:,2], 0, 255).astype(np.uint8)
            recreated_ycbcr = np.dstack((recreated_y, recreated_cb, recreated_cr))
            recreated = cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)
        elif method == 'ssim':
            ssim_diff = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_GRAYSCALE)
            ssim_map = ssim_diff / 255.0
            
            # Make the blending more aggressive
            alpha = np.power(ssim_map, 0.5)  # This will make the transition sharper
            recreated = ((1 - alpha[:,:,np.newaxis]) * earlier_screenshot + alpha[:,:,np.newaxis] * next_screenshot).astype(np.uint8)
            
        if np.array_equal(recreated, next_screenshot):
            perfect_recreations += 1
        
        cv2.imwrite(os.path.join(recreated_folder, f"recreated_{i+1:04d}.png"), recreated)
    
    print(f"Recreated {total_recreations} screenshots.")
    print(f"Perfect recreations: {perfect_recreations}/{total_recreations} ({perfect_recreations/total_recreations*100:.2f}%)")
    print(f"===== EXITING recreate_screenshots ({method}) FUNCTION =====")

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
    for frame, mse in high_mse_frames[:10]:  # Print details for the top 10 high MSE frames
        print(f"Frame {frame}: MSE = {mse:.2f}")
        
        # Analyze the differences
        original = cv2.imread(os.path.join(screenshots_folder, screenshots[frame]))
        recreation = cv2.imread(os.path.join(recreated_folder, recreated[frame-1]))
        diff = cv2.absdiff(original, recreation)
        
        # Calculate the percentage of changed pixels
        changed_pixels = np.sum(diff > 0) / (diff.shape[0] * diff.shape[1] * diff.shape[2]) * 100
        print(f"  Changed pixels: {changed_pixels:.2f}%")
        
        # Analyze color channel differences
        for c, color in enumerate(['Blue', 'Green', 'Red']):
            channel_diff = diff[:,:,c]
            channel_changed = np.sum(channel_diff > 0) / (channel_diff.shape[0] * channel_diff.shape[1]) * 100
            print(f"  {color} channel changes: {channel_changed:.2f}%")
    
    print("===== EXITING analyze_high_mse_frames FUNCTION =====")

def main():
    print("===== ENTERING main FUNCTION =====")
    try:
        screenshots_folder = os.path.join("screenshots", "originals")
        diffs_folder = os.path.join("screenshots", "diffs")
        generate_diffs(screenshots_folder, diffs_folder)
        
        methods = ['current', 'ycbcr', 'combined', 'ssim']
        for method in methods:
            print(f"\nTesting {method} method:")
            delta_folder = os.path.join(diffs_folder, f"delta_{method if method != 'current' else 'diffs'}")
            recreated_folder = os.path.join("screenshots", "recreations", method)
            recreate_screenshots(screenshots_folder, delta_folder, recreated_folder, method)
            verify_recreation(screenshots_folder, recreated_folder)
            
            # Analyze high MSE frames for each method
            print(f"\nAnalyzing high MSE frames for {method} method:")
            analyze_high_mse_frames(screenshots_folder, recreated_folder)
    except Exception as e:
        print(f"===== ERROR IN main: {str(e)} =====")
    print("===== EXITING main FUNCTION =====")
    
if __name__ == "__main__":
    print("===== SCRIPT STARTED =====")
    main()
    print("===== SCRIPT ENDED =====")

# Keep console open
input("Press Enter to exit...")
