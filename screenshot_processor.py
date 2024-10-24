import os
import cv2
import numpy as np
import json
from config import *
from diff_methods import METHOD_DICT
from image_utils import verify_recreation, verify_reversed_recreation, analyze_high_mse_frames

def generate_diffs(screenshots_folder, diffs_folder):
    print("===== ENTERING generate_diffs FUNCTION =====")
    screenshots = sorted([f for f in os.listdir(screenshots_folder) if f.endswith('.png')])
    
    for method_name, method in METHOD_DICT.items():
        print(f"Processing {method_name}... (Conflict policy: {method.config['diff']})")
        method_diffs_folder = os.path.join(diffs_folder, method_name)
        method_recreations_folder = os.path.join(RECREATIONS_FOLDER, method_name)
        method_recreations_reversed_folder = os.path.join(RECREATIONS_FOLDER, f"{method_name}_reversed")
        
        os.makedirs(method_diffs_folder, exist_ok=True)
        os.makedirs(method_recreations_folder, exist_ok=True)
        os.makedirs(method_recreations_reversed_folder, exist_ok=True)
        
        # Check if the last diff file exists as a shortcut for skipping
        last_diff_file = os.path.join(method_diffs_folder, f"diff_{len(screenshots)-2:04d}.png")
        if os.path.exists(last_diff_file) and method.config['diff'] == 'skip':
            print(f"Skipping entire batch for {method_name} (last diff file exists)")
            continue
        
        for i in range(len(screenshots) - 1):
            diff_file = os.path.join(method_diffs_folder, f"diff_{i:04d}.png")
            
            # Skip if the diff file exists and analysis is set to 'skip'
            if os.path.exists(diff_file) and method.config['analysis'] == 'skip':
                continue
            
            img1 = cv2.imread(os.path.join(screenshots_folder, screenshots[i]))
            img2 = cv2.imread(os.path.join(screenshots_folder, screenshots[i+1]))
            
            diff = method.generate_diff(img1, img2)
            cv2.imwrite(diff_file, diff)
            
            recreated = method.recreate_screenshot(img1, diff)
            cv2.imwrite(os.path.join(method_recreations_folder, f"recreated_{i+1:04d}.png"), recreated)
            
            recreated_reversed = method.recreate_previous_screenshot(img2, diff)
            cv2.imwrite(os.path.join(method_recreations_reversed_folder, f"recreated_reversed_{i:04d}.png"), recreated_reversed)
    
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
        analyses_folder = os.path.join('screenshots', 'analyses')
        os.makedirs(analyses_folder, exist_ok=True)

        generate_diffs(SCREENSHOTS_FOLDER, DIFFS_FOLDER)
        
        summary = []
        summary_tuned = []

        for method_name, method in METHOD_DICT.items():
            print(f"\nProcessing {method_name} method:")
            delta_folder = os.path.join(DIFFS_FOLDER, f"delta_{method_name}")
            recreated_folder = os.path.join(RECREATIONS_FOLDER, method_name)
            recreated_reversed_folder = os.path.join(RECREATIONS_FOLDER, f"{method_name}_reversed")
            
            recreate_screenshots(SCREENSHOTS_FOLDER, delta_folder, recreated_folder, method_name)
            
            analysis_file = os.path.join(analyses_folder, f"{method_name}.json")
            analysis_cache = {}
            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r') as f:
                        analysis_cache = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON from {analysis_file}: {str(e)}")
                    print("Continuing with empty analysis cache.")

            if method.config['analysis'] == 'skip' and 'verification' in analysis_cache and 'verification_reversed' in analysis_cache and 'high_mse_analysis' in analysis_cache:
                print(f"Skipping analysis for {method_name} (using cached results)")
                verification_results = analysis_cache['verification']
                verification_reversed_results = analysis_cache['verification_reversed']
                high_mse_analysis = analysis_cache['high_mse_analysis']
            else:
                print(f"Verifying recreation for {method_name} method:")
                verification_results = verify_recreation(SCREENSHOTS_FOLDER, recreated_folder, method_name)
                
                print(f"Verifying reversed recreation for {method_name} method:")
                verification_reversed_results = verify_reversed_recreation(SCREENSHOTS_FOLDER, recreated_reversed_folder, method_name)
                
                print(f"\nAnalyzing high MSE frames for {method_name} method:")
                cached_high_mse = analysis_cache.get('high_mse_analysis')
                high_mse_analysis = analyze_high_mse_frames(SCREENSHOTS_FOLDER, recreated_folder, cached_result=cached_high_mse)
                
                analysis_cache.update({
                    'verification': verification_results,
                    'verification_reversed': verification_reversed_results,
                    'high_mse_analysis': high_mse_analysis
                })

                try:
                    with open(analysis_file, 'w') as f:
                        json.dump(analysis_cache, f, indent=2, default=float)
                except TypeError as e:
                    print(f"Error writing JSON to {analysis_file}: {str(e)}")
                    print("Attempting to serialize with default=str...")
                    try:
                        with open(analysis_file, 'w') as f:
                            json.dump(analysis_cache, f, indent=2, default=str)
                    except Exception as e:
                        print(f"Failed to serialize with default=str: {str(e)}")

            summary_entry = generate_summary_entry(method_name, verification_results, verification_reversed_results, high_mse_analysis)
            summary.append(summary_entry)

            if method.config['tune']:
                summary_tuned.append(summary_entry)

        try:
            write_summary('summary.txt', summary)
            write_summary('summary_tuned.txt', summary_tuned)
        except TypeError as e:
            print(f"Error during JSON serialization: {str(e)}")
            print("Attempting to serialize with default=str...")
            try:
                with open('summary.txt', 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                with open('summary_tuned.txt', 'w') as f:
                    json.dump(summary_tuned, f, indent=2, default=str)
            except Exception as e:
                print(f"Failed to serialize with default=str: {str(e)}")

    except Exception as e:
        print(f"===== ERROR IN process_screenshots: {str(e)} =====")
        import traceback
        traceback.print_exc()
    print("===== EXITING process_screenshots FUNCTION =====")

def generate_summary_entry(method_name, verification_results, verification_reversed_results, high_mse_analysis):
    summary = f"Method: {method_name}\n"
    
    # Forward verification results
    if verification_results:
        summary += f"Forward - Average MSE: {float(verification_results.get('avg_mse', 0)):.2f}\n"
        summary += f"Forward - Average SSIM: {verification_results.get('avg_ssim', 0):.4f}\n"
        total_comparisons = verification_results.get('total_comparisons', 1)
        perfect_matches = verification_results.get('perfect_matches', 0)
        percentage = (perfect_matches / total_comparisons * 100) if total_comparisons > 0 else 0
        summary += f"Forward - Perfect matches: {perfect_matches}/{total_comparisons} ({percentage:.2f}%)\n"
    else:
        summary += "Forward verification results not available\n"
    
    # Reverse verification results
    if verification_reversed_results:
        summary += f"Reverse - Average MSE: {float(verification_reversed_results.get('avg_mse', 0)):.2f}\n"
        summary += f"Reverse - Average SSIM: {verification_reversed_results.get('avg_ssim', 0):.4f}\n"
        total_comparisons = verification_reversed_results.get('total_comparisons', 1)
        perfect_matches = verification_reversed_results.get('perfect_matches', 0)
        percentage = (perfect_matches / total_comparisons * 100) if total_comparisons > 0 else 0
        summary += f"Reverse - Perfect matches: {perfect_matches}/{total_comparisons} ({percentage:.2f}%)\n"
    else:
        summary += "Reverse verification results not available\n"
    
    # High MSE analysis
    if high_mse_analysis:
        summary += f"High MSE frames: {high_mse_analysis.get('high_mse_count', 0)} (threshold: {high_mse_analysis.get('threshold', 0)})\n"
        
        if high_mse_analysis.get('detailed_analysis'):
            summary += "Top 3 high MSE frames:\n"
            for frame in high_mse_analysis['detailed_analysis'][:3]:
                summary += f"  Frame {frame.get('frame', 'N/A')}: MSE = {float(frame.get('mse', 0)):.2f}, Changed pixels: {float(frame.get('changed_pixels', 0)):.2f}%\n"
    else:
        summary += "High MSE analysis not available\n"
    
    return summary

def write_summary(filename, summary_entries):
    with open(os.path.join('screenshots', 'analyses', filename), 'w') as f:
        f.write("\n\n".join(summary_entries))
    print(f"Summary written to {filename}")
