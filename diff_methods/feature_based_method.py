import cv2
import numpy as np
from .base_method import BaseDiffMethod

class FeatureBasedMethod(BaseDiffMethod):
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None

    def generate_diff(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        self.keypoints1, self.descriptors1 = self.orb.detectAndCompute(gray1, None)
        self.keypoints2, self.descriptors2 = self.orb.detectAndCompute(gray2, None)
        
        matches = self.bf.match(self.descriptors1, self.descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        diff = np.zeros_like(img1)
        for match in matches:
            pt1 = tuple(map(int, self.keypoints1[match.queryIdx].pt))
            pt2 = tuple(map(int, self.keypoints2[match.trainIdx].pt))
            cv2.line(diff, pt1, pt2, (0, 255, 0), 1)
        
        return diff

    def recreate_screenshot(self, earlier_screenshot, delta):
        if self.keypoints1 is None or self.keypoints2 is None or self.descriptors1 is None or self.descriptors2 is None:
            # If keypoints or descriptors are not available, return the earlier screenshot
            return earlier_screenshot

        h, w = earlier_screenshot.shape[:2]
        recreated = np.zeros_like(earlier_screenshot)
        
        matches = self.bf.match(self.descriptors1, self.descriptors2)
        
        for match in matches:
            pt1 = tuple(map(int, self.keypoints1[match.queryIdx].pt))
            pt2 = tuple(map(int, self.keypoints2[match.trainIdx].pt))
            if 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                recreated[pt2[1], pt2[0]] = earlier_screenshot[pt1[1], pt1[0]]
        
        # Fill in gaps
        mask = np.all(recreated == 0, axis=2)
        recreated[mask] = earlier_screenshot[mask]
        
        return recreated

    def recreate_previous_screenshot(self, later_screenshot, delta):
        return self.recreate_screenshot(later_screenshot, self.reverse_diff(delta))

    def reverse_diff(self, delta):
        reversed_diff = np.zeros_like(delta)
        mask = delta[:,:,3]  # Alpha channel
        reversed_diff[:,:,3] = mask  # Keep the same mask

        matches = self.bf.match(self.descriptors2, self.descriptors1)  # Note the order change
        
        for match in matches:
            pt2 = tuple(map(int, self.keypoints2[match.queryIdx].pt))
            pt1 = tuple(map(int, self.keypoints1[match.trainIdx].pt))
            if mask[pt1[1], pt1[0]] > 0:
                reversed_diff[pt1[1], pt1[0], :3] = delta[pt2[1], pt2[0], :3]

        return reversed_diff

    @property
    def name(self):
        return 'feature_based'

