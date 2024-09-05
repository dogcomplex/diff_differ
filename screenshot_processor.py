import os
import cv2
import numpy as np
import json
from config import *
from diff_methods import METHOD_DICT
from image_utils import verify_recreation, analyze_high_mse_frames

def generate_diffs(screenshots_folder, diffs_folder):
    print("===== ENTERING generate_diffs FUNCTION =====")
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    
    if len(screenshots) < 2:
        print("Not enough screenshots to generate diffs.")
        return

    last_screenshot_pair = (screenshots[-2], screenshots[-1])
    
    for method_name, method in METHOD_DICT.items():
        delta_folder = os.path.join(diffs_folder, f"delta_{method_name}")
        os.makedirs(delta_folder, exist_ok=True)
        
        # Check if the last diff file exists
        last_diff_filename = f"diff_{method_name}_{len(screenshots)-2:04d}_{len(screenshots)-1:04d}.png"
        last_diff_path = os.path.join(delta_folder, last_diff_filename)
        
        if os.path.exists(last_diff_path) and method.config['diff'] != 'overwrite':
            print(f"Skipping diff generation for {method_name} (last diff file exists)")
            continue
        
        print(f"Now generating diffs for: {method_name}")
        
        for i in range(len(screenshots) - 1):
            img1 = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
            img2 = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
            
            if img1 is None or img2 is None or img1.shape != img2.shape:
                print(f"Error processing pair {i} and {i+1}")
                continue
            
            output_path = os.path.join(delta_folder, f"diff_{method_name}_{i:04d}_{i+1:04d}.png")
            
            if not os.path.exists(output_path) or method.config['diff'] == 'overwrite':
                delta = method.generate_diff(img1, img2)
                cv2.imwrite(output_path, delta)
    
    print("===== EXITING generate_diffs FUNCTION =====")

def recreate_screenshots(screenshots_folder, delta_folder, recreated_folder, method_name):
    print(f"===== ENTERING recreate_screenshots ({method_name}) FUNCTION =====")
    method = METHOD_DICT[method_name]
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    delta_files = sorted([f for f in os.listdir(delta_folder) if f.endswith('.png')])
    
    os.makedirs(recreated_folder, exist_ok=True)
    
    # Check if the last recreated file exists
    last_recreated_filename = f"recreated_{len(delta_files):04d}.png"
    last_recreated_path = os.path.join(recreated_folder, last_recreated_filename)
    
    if os.path.exists(last_recreated_path) and method.config['recreation'] != 'overwrite':
        print(f"Skipping recreation for {method_name} (last recreated file exists)")
        return
    
    for i in range(len(delta_files)):
        earlier_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        delta = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_UNCHANGED)
        
        output_path = os.path.join(recreated_folder, f"recreated_{i+1:04d}.png")
        
        if not os.path.exists(output_path) or method.config['recreation'] == 'overwrite':
            recreated = method.recreate_screenshot(earlier_screenshot, delta)
            cv2.imwrite(output_path, recreated)
    
    print(f"===== EXITING recreate_screenshots ({method_name}) FUNCTION =====")

def process_screenshots():
    print("===== ENTERING process_screenshots FUNCTION =====")
    try:
        cache_file = 'analysis_cache.json'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {cache_file} contains invalid JSON. Starting with an empty cache.")
                cache = {}
        else:
            cache = {}

        generate_diffs(SCREENSHOTS_FOLDER, DIFFS_FOLDER)
        
        for method_name, method in METHOD_DICT.items():
            print(f"\nProcessing {method_name} method:")
            delta_folder = os.path.join(DIFFS_FOLDER, f"delta_{method_name}")
            recreated_folder = os.path.join(RECREATIONS_FOLDER, method_name)
            
            recreate_screenshots(SCREENSHOTS_FOLDER, delta_folder, recreated_folder, method_name)
            
            if method.config['analysis'] == 'skip' and method_name in cache:
                print(f"Skipping analysis for {method_name} (using cached results)")
                continue
            
            print(f"Verifying recreation for {method_name} method:")
            verification_results = verify_recreation(SCREENSHOTS_FOLDER, recreated_folder)
            
            print(f"\nAnalyzing high MSE frames for {method_name} method:")
            high_mse_analysis = analyze_high_mse_frames(SCREENSHOTS_FOLDER, recreated_folder)
            
            cache[method_name] = {
                'verification': verification_results,
                'high_mse_analysis': high_mse_analysis
            }
        
        with open(cache_file, 'w') as f:
            json.dump(cache, f)

    except Exception as e:
        print(f"===== ERROR IN process_screenshots: {str(e)} =====")
    print("===== EXITING process_screenshots FUNCTION =====")