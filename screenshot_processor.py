import os
import cv2
import numpy as np
from config import *
from diff_methods import METHOD_DICT
from image_utils import verify_recreation, analyze_high_mse_frames

def generate_diffs(screenshots_folder, diffs_folder):
    print("===== ENTERING generate_diffs FUNCTION =====")
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    
    for method_name, method in METHOD_DICT.items():
        delta_folder = os.path.join(diffs_folder, f"delta_{method_name}")
        os.makedirs(delta_folder, exist_ok=True)
        
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
    
    for i in range(len(delta_files)):
        earlier_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
        delta = cv2.imread(os.path.join(delta_folder, delta_files[i]), cv2.IMREAD_UNCHANGED)
        next_screenshot = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
        
        output_path = os.path.join(recreated_folder, f"recreated_{i+1:04d}.png")
        
        if not os.path.exists(output_path) or method.config['recreation'] == 'overwrite':
            recreated = method.recreate_screenshot(earlier_screenshot, delta, next_screenshot)
            cv2.imwrite(output_path, recreated)
    
    print(f"===== EXITING recreate_screenshots ({method_name}) FUNCTION =====")

def process_screenshots():
    print("===== ENTERING process_screenshots FUNCTION =====")
    try:
        generate_diffs(SCREENSHOTS_FOLDER, DIFFS_FOLDER)
        
        for method_name in METHOD_DICT.keys():
            print(f"\nTesting {method_name} method:")
            delta_folder = os.path.join(DIFFS_FOLDER, f"delta_{method_name}")
            recreated_folder = os.path.join(RECREATIONS_FOLDER, method_name)
            recreate_screenshots(SCREENSHOTS_FOLDER, delta_folder, recreated_folder, method_name)
            verify_recreation(SCREENSHOTS_FOLDER, recreated_folder)
            
            print(f"\nAnalyzing high MSE frames for {method_name} method:")
            analyze_high_mse_frames(SCREENSHOTS_FOLDER, recreated_folder)
    except Exception as e:
        print(f"===== ERROR IN process_screenshots: {str(e)} =====")
    print("===== EXITING process_screenshots FUNCTION =====")