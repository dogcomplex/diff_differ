print("===== SCRIPT STARTED =====")

try:
    import cv2
    import os
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.segmentation import slic
    from skimage.util import img_as_float
    import time
    import signal
    from scipy import ndimage
    from skimage import filters, segmentation, morphology

    # Configuration
    class Config:
        DEFAULT_FILE_CONFLICT_BEHAVIOR = 'skip'  # Global default
        METHOD_FILE_CONFLICT_BEHAVIOR = {
            'current': {'diff': 'skip', 'recreation': 'skip'},
            'ycbcr': {'diff': 'skip', 'recreation': 'skip'},
            'combined': {'diff': 'skip', 'recreation': 'skip'},
            'ssim': {'diff': 'overwrite', 'recreation': 'overwrite'},
            'superpixel': {'diff': 'skip', 'recreation': 'skip'},
            'membrane': {'diff': 'overwrite', 'recreation': 'overwrite'}
        }

    # You can modify these parameters as needed
    SSIM_BETA = 4.0
    SSIM_THRESHOLD = 0.2
    SUPERPIXEL_SEGMENTS = 2000  # Increased from 500
    SUPERPIXEL_COMPACTNESS = 5  # Decreased from 20
    SUPERPIXEL_CHANGE_THRESHOLD = 0.01  # Reduced from 0.1
    SUPERPIXEL_TIMEOUT = 30  # Timeout in seconds

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Function call timed out")

    def generate_superpixels(img, n_segments, compactness):
        return slic(img, n_segments=n_segments, compactness=compactness)

    
    def generate_superpixel_diff_with_timeout(img1, img2):
        img1_float = img_as_float(img1)
        img2_float = img_as_float(img2)

        segments1 = generate_superpixels(img1_float, SUPERPIXEL_SEGMENTS, SUPERPIXEL_COMPACTNESS)
        segments2 = generate_superpixels(img2_float, SUPERPIXEL_SEGMENTS, SUPERPIXEL_COMPACTNESS)

        # Initialize diff image
        diff = np.zeros_like(img1, dtype=np.float32)
        
        # Compare corresponding superpixels
        for segment in np.unique(segments1):
            mask1 = segments1 == segment
            
            # Find the most overlapping segment in the second image
            overlap = np.bincount(segments2[mask1].ravel())
            best_segment2 = overlap.argmax()
            mask2 = segments2 == best_segment2
            
            # Combine masks to ensure we're comparing the same areas
            combined_mask = mask1 & mask2
            
            # Calculate the difference for this superpixel
            superpixel_diff = img2_float[combined_mask] - img1_float[combined_mask]
            
            # Calculate the mean absolute difference for each channel
            mean_abs_diff = np.mean(np.abs(superpixel_diff), axis=0)
            
            # If any channel has changed significantly, add the entire superpixel diff
            if np.any(mean_abs_diff > SUPERPIXEL_CHANGE_THRESHOLD):
                diff[combined_mask] = superpixel_diff
        
        # Normalize and convert to uint8
        diff = np.clip(diff * 127.5 + 127.5, 0, 255).astype(np.uint8)
        return diff

    def generate_membrane_diff(img1, img2, edge_threshold=0.1, region_threshold=0.05):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Compute edge difference
        edges1 = filters.sobel(gray1)
        edges2 = filters.sobel(gray2)
        edge_diff = np.abs(edges2 - edges1)
        
        # Create initial mask based on edge differences
        mask = edge_diff > edge_threshold
        
        # Dilate the mask to connect nearby edges
        mask = morphology.dilation(mask, morphology.disk(3))
        
        # Use the watershed algorithm to grow regions
        distance = ndimage.distance_transform_edt(~mask)  # Changed this line
        markers = filters.threshold_local(distance, 31, offset=0.1)
        markers = morphology.label(markers)
        
        # Segment the image
        segments = segmentation.watershed(-distance, markers, mask=mask)
        
        # Compute the difference for each segment
        diff = np.zeros_like(img1, dtype=np.float32)
        for segment in np.unique(segments):
            if segment == 0:  # background
                continue
            segment_mask = segments == segment
            segment_diff = img2[segment_mask].astype(np.float32) - img1[segment_mask].astype(np.float32)
            if np.mean(np.abs(segment_diff)) > region_threshold:
                diff[segment_mask] = segment_diff
        
        # Normalize and convert to uint8
        diff = np.clip(diff * 127.5 + 127.5, 0, 255).astype(np.uint8)
        return diff

    def generate_diffs(screenshots_folder, diffs_folder):
        print("===== ENTERING generate_diffs FUNCTION =====")
        delta_diffs_folder = os.path.join(diffs_folder, "delta_diffs")
        delta_ycbcr_folder = os.path.join(diffs_folder, "delta_ycbcr")
        delta_combined_folder = os.path.join(diffs_folder, "delta_combined")
        delta_ssim_folder = os.path.join(diffs_folder, "delta_ssim")
        delta_superpixel_folder = os.path.join(diffs_folder, "delta_superpixel")
        delta_membrane_folder = os.path.join(diffs_folder, "delta_membrane")
        
        os.makedirs(delta_diffs_folder, exist_ok=True)
        os.makedirs(delta_ycbcr_folder, exist_ok=True)
        os.makedirs(delta_combined_folder, exist_ok=True)
        os.makedirs(delta_ssim_folder, exist_ok=True)
        os.makedirs(delta_superpixel_folder, exist_ok=True)
        os.makedirs(delta_membrane_folder, exist_ok=True)
        
        screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])

        total_screenshots = len(screenshots) - 1
        for i in range(total_screenshots):
            try:
                print(f"Processing pair {i+1}/{total_screenshots}", end='\r')
                img1 = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
                img2 = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
                
                if img1 is None or img2 is None:
                    print(f"\nError reading images for pair {i} and {i+1}")
                    continue
                
                if img1.shape != img2.shape:
                    print(f"\nImage shapes do not match for pair {i} and {i+1}")
                    continue
                
                for method in ['current', 'ycbcr', 'combined', 'ssim', 'superpixel', 'membrane']:
                    output_folder = os.path.join(diffs_folder, f"delta_{method if method != 'current' else 'diffs'}")
                    output_path = os.path.join(output_folder, f"diff_{method}_{i:04d}_{i+1:04d}.png")
                    
                    if not os.path.exists(output_path) or Config.METHOD_FILE_CONFLICT_BEHAVIOR[method]['diff'] == 'overwrite':
                        if method == 'current':
                            delta = cv2.subtract(img2.astype(np.int16), img1.astype(np.int16))
                            cv2.imwrite(output_path, delta)
                        elif method == 'ycbcr':
                            img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
                            img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
                            delta_ycbcr = cv2.subtract(img2_ycbcr.astype(np.int16), img1_ycbcr.astype(np.int16))
                            cv2.imwrite(output_path, delta_ycbcr)
                        elif method == 'combined':
                            img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
                            img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
                            delta_y = cv2.subtract(img2[:,:,0].astype(np.int16), img1[:,:,0].astype(np.int16))
                            delta_cb = cv2.subtract(img2_ycbcr[:,:,1].astype(np.int16), img1_ycbcr[:,:,1].astype(np.int16))
                            delta_cr = cv2.subtract(img2_ycbcr[:,:,2].astype(np.int16), img1_ycbcr[:,:,2].astype(np.int16))
                            delta_combined = np.dstack((delta_y, delta_cb, delta_cr))
                            cv2.imwrite(output_path, delta_combined)
                        elif method == 'ssim':
                            ssim_diff = generate_ssim_diff(img1, img2, beta=SSIM_BETA, threshold=SSIM_THRESHOLD)
                            if ssim_diff.shape != img1.shape:
                                print(f"Error: SSIM diff shape mismatch for pair {i} and {i+1}")
                                continue
                            cv2.imwrite(output_path, ssim_diff)
                        elif method == 'superpixel':
                            start_time = time.time()
                            superpixel_diff = generate_superpixel_diff_with_timeout(img1, img2)
                            cv2.imwrite(output_path, superpixel_diff)
                            print(f"\nSuperpixel diff for pair {i} and {i+1} took {time.time() - start_time:.2f} seconds")
                        elif method == 'membrane':
                            start_time = time.time()
                            membrane_diff = generate_membrane_diff(img1, img2)
                            cv2.imwrite(output_path, membrane_diff)
                            print(f"\nMembrane diff for pair {i} and {i+1} took {time.time() - start_time:.2f} seconds")
                    else:
                        #print(f"Skipping {method} diff for pair {i} and {i+1} (file exists)")
                        pass
            except Exception as e:
                print(f"\nError processing pair {i} and {i+1}: {str(e)}")

        print(f"\nGenerated diff images for {total_screenshots} pairs of screenshots.")
        print("===== EXITING generate_diffs FUNCTION =====")

    def generate_ssim_diff(img1, img2, beta=SSIM_BETA, threshold=SSIM_THRESHOLD):
        ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
        
        # Apply a non-linear transformation to make the diff more aggressive
        ssim_diff = np.power(1 - ssim_map, beta)
        
        # Apply a threshold to further emphasize changes
        ssim_diff[ssim_diff < threshold] = 0
        
        # Ensure the diff is in the correct range and data type
        ssim_diff = (ssim_diff * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to 3-channel image if it's single channel
        if ssim_diff.ndim == 2:
            ssim_diff = np.stack([ssim_diff] * 3, axis=-1)
        
        return ssim_diff

    def recreate_screenshots(screenshots_folder, delta_folder, recreated_folder, method):
        print(f"===== ENTERING recreate_screenshots ({method}) FUNCTION =====")
        os.makedirs(recreated_folder, exist_ok=True)
        
        screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
        delta_files = sorted([f for f in os.listdir(delta_folder) if f.endswith('.png')])
        
        perfect_recreations = 0
        total_recreations = len(delta_files)
        
        for i in range(len(delta_files)):
            output_path = os.path.join(recreated_folder, f"recreated_{i+1:04d}.png")
            if os.path.exists(output_path) and Config.METHOD_FILE_CONFLICT_BEHAVIOR[method]['recreation'] == 'skip':
                # print(f"Skipping recreation for {method} method, frame {i+1} (file exists)")
                continue

            earlier_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
            
            # Check if we're at the last screenshot
            if i + 1 < len(screenshots):
                next_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
            else:
                print(f"Reached the last screenshot. Skipping recreation for frame {i+1}")
                break

            delta = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_UNCHANGED)
            
            if method == 'current':
                recreated = np.clip(earlier_screenshot.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
            elif method == 'ycbcr':
                earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
                recreated_ycbcr = np.clip(earlier_ycbcr.astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
                recreated = cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)
            elif method == 'combined':
                recreated_y = np.clip(earlier_screenshot[:,:,0].astype(np.int16) + delta[:,:,0].astype(np.int16), 0, 255).astype(np.uint8)
                earlier_ycbcr = cv2.cvtColor(earlier_screenshot, cv2.COLOR_BGR2YCrCb)
                recreated_cb = np.clip(earlier_ycbcr[:,:,1].astype(np.int16) + delta[:,:,1].astype(np.int16), 0, 255).astype(np.uint8)
                recreated_cr = np.clip(earlier_ycbcr[:,:,2].astype(np.int16) + delta[:,:,2].astype(np.int16), 0, 255).astype(np.uint8)
                recreated_ycbcr = np.dstack((recreated_y, recreated_cb, recreated_cr))
                recreated = cv2.cvtColor(recreated_ycbcr, cv2.COLOR_YCrCb2BGR)
            elif method == 'ssim':
                if delta is None:
                    print(f"Error reading SSIM delta for frame {i+1}")
                    continue
                if delta.shape != earlier_screenshot.shape:
                    print(f"SSIM delta shape mismatch for frame {i+1}")
                    continue
                ssim_map = delta.astype(np.float32) / 255.0
                alpha = np.power(ssim_map, 0.25)
                recreated = ((1 - alpha) * earlier_screenshot + alpha * next_screenshot).clip(0, 255).astype(np.uint8)
            elif method in ['superpixel', 'membrane']:
                mask = np.any(np.abs(delta.astype(np.int16) - 127) > 1, axis=-1)
                recreated = earlier_screenshot.copy()
                recreated[mask] = np.clip(earlier_screenshot[mask].astype(np.float32) + (delta[mask].astype(np.float32) - 127.5) * 2, 0, 255).astype(np.uint8)
            
            if np.array_equal(recreated, next_screenshot):
                perfect_recreations += 1
            
            cv2.imwrite(output_path, recreated)

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
            
            methods = ['current', 'ycbcr', 'combined', 'ssim', 'superpixel', 'membrane']
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
except ImportError as e:
    print(f"===== IMPORT ERROR: {e} =====")
