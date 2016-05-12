from .gcm_dataset import GCMDataset

class GFDLDataset(GCMDataset):
    _vertical_coords = {'pfull', 'phalf'}
    _horiz_coords = {'lat', 'latb', 'lon', 'lonb'}
