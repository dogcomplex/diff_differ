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

# Constants
SSIM_BETA = 4.0
SSIM_THRESHOLD = 0.2
SUPERPIXEL_SEGMENTS = 2000
SUPERPIXEL_COMPACTNESS = 5
SUPERPIXEL_CHANGE_THRESHOLD = 0.01
SUPERPIXEL_TIMEOUT = 30

# Folder paths
SCREENSHOTS_FOLDER = "screenshots/originals"
DIFFS_FOLDER = "screenshots/diffs"
RECREATIONS_FOLDER = "screenshots/recreations"

# Available methods
METHODS = ['current', 'ycbcr', 'combined', 'ssim', 'superpixel', 'membrane']
