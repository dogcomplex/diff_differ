from abc import ABC, abstractmethod

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
            'analysis': 'skip'
        }
