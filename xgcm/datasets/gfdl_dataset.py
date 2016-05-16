from .gcm_dataset import GCMDataset



class GFDLDataset(GCMDataset):

    _needed_vars = ['phalf', 'pfull', 'lat', 'lon', 'latb', 'lonb']

    _coord_map = {
        'z_centre': 'phalf',
        'z_edge': 'pfull',
        'x_centre': 'lon',
        'x_edge': 'lonb',
        'y_centre': 'lat',
        'y_edge': 'latb'
    }
    _x_periodic = True
    _vertical_coords = {'pfull', 'phalf'}
    _horiz_coords = {'lat', 'latb', 'lon', 'lonb'}
    _centre_coords = {'x': 'lon', 'y': 'lat'}
    _edge_coords = {'x': 'lonb', 'y': 'latb'}