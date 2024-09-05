import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy import ndimage
from skimage import filters, segmentation, morphology
from config import *

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

    diff = np.zeros_like(img1, dtype=np.float32)
    
    for segment in np.unique(segments1):
        mask1 = segments1 == segment
        overlap = np.bincount(segments2[mask1].ravel())
        best_segment2 = overlap.argmax()
        mask2 = segments2 == best_segment2
        combined_mask = mask1 & mask2
        superpixel_diff = img2_float[combined_mask] - img1_float[combined_mask]
        mean_abs_diff = np.mean(np.abs(superpixel_diff), axis=0)
        
        if np.any(mean_abs_diff > SUPERPIXEL_CHANGE_THRESHOLD):
            diff[combined_mask] = superpixel_diff
    
    diff = np.clip(diff * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return diff

def generate_membrane_diff(img1, img2, edge_threshold=0.1, region_threshold=0.05):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    edges1 = filters.sobel(gray1)
    edges2 = filters.sobel(gray2)
    edge_diff = np.abs(edges2 - edges1)
    
    mask = edge_diff > edge_threshold
    mask = morphology.dilation(mask, morphology.disk(3))
    
    distance = ndimage.distance_transform_edt(~mask)
    markers = filters.threshold_local(distance, 31, offset=0.1)
    markers = morphology.label(markers)
    
    segments = segmentation.watershed(-distance, markers, mask=mask)
    
    diff = np.zeros_like(img1, dtype=np.float32)
    for segment in np.unique(segments):
        if segment == 0:
            continue
        segment_mask = segments == segment
        segment_diff = img2[segment_mask].astype(np.float32) - img1[segment_mask].astype(np.float32)
        if np.mean(np.abs(segment_diff)) > region_threshold:
            diff[segment_mask] = segment_diff
    
    diff = np.clip(diff * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return diff

def generate_ssim_diff(img1, img2, beta=SSIM_BETA, threshold=SSIM_THRESHOLD):
    ssim_map = ssim(img1, img2, win_size=3, channel_axis=2, full=True)[1]
    ssim_diff = np.power(1 - ssim_map, beta)
    ssim_diff[ssim_diff < threshold] = 0
    ssim_diff = (ssim_diff * 255).clip(0, 255).astype(np.uint8)
    
    if ssim_diff.ndim == 2:
        ssim_diff = np.stack([ssim_diff] * 3, axis=-1)
    
    return ssim_diff

def generate_current_diff(img1, img2):
    return cv2.subtract(img2.astype(np.int16), img1.astype(np.int16))

def generate_ycbcr_diff(img1, img2):
    img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
    return cv2.subtract(img2_ycbcr.astype(np.int16), img1_ycbcr.astype(np.int16))

def generate_combined_diff(img1, img2):
    img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
    delta_y = cv2.subtract(img2[:,:,0].astype(np.int16), img1[:,:,0].astype(np.int16))
    delta_cb = cv2.subtract(img2_ycbcr[:,:,1].astype(np.int16), img1_ycbcr[:,:,1].astype(np.int16))
    delta_cr = cv2.subtract(img2_ycbcr[:,:,2].astype(np.int16), img1_ycbcr[:,:,2].astype(np.int16))
    return np.dstack((delta_y, delta_cb, delta_cr))
