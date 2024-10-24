from abc import ABC, abstractmethod
import numpy as np

class BaseDiffMethod(ABC):
    @abstractmethod
    def generate_diff(self, img1, img2):
        pass

    @abstractmethod
    def recreate_screenshot(self, earlier_screenshot, delta):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def config(self):
        return {
            'diff': 'skip',
            'recreation': 'skip',
            'analysis': 'skip',
            'tune': False
        }

    def reverse_diff(self, delta):
        # Default implementation: invert the delta
        if delta.dtype == np.uint8:
            return 255 - delta
        else:
            return -delta

    def recreate_previous_screenshot(self, later_screenshot, delta):
        return self.recreate_screenshot(later_screenshot, self.reverse_diff(delta))
