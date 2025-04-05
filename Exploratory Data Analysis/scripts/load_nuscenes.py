from nuscenes.nuscenes import NuScenes
from config import DATA_ROOT, VERSION

def load_nuscenes():
    return NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)