gfdl = {
    'lat': {
        'name': 'Latitude',
        'units': 'degrees N',
        'label': 'y_centre',
        'periodic': False,
    },
    'latb': {
        'name': 'Latitude (edges)',
        'units': 'degrees N',
        'label': 'y_edge',
        'periodic': False
    },
    'lon': {
        'name': 'Longitude',
        'units': 'degrees E',
        'label': 'x_centre',
        'periodic': True
#        'diff_fn': _dlon,
    },
    'lonb': {
        'name': 'Longitude (edges)',
        'units': 'degrees E',
        'label': 'x_centre',
        'periodic': True
#        'diff_fn': _dlon,
    },
    'pfull': {
        'name': 'Pressure',
        'units': 'hPa',
        'label': 'z_centre',
        'periodic': False
    },
    'phalf': {
        'name': 'Pressure (edges)',
        'units': 'hPa',
        'label': 'z_edge',
        'periodic': False
    },
}


mitgcm = {
    'Y': {
        'name': 'Y position',
        'units': 'm',
        'label': 'y_centre',
        'periodic': False
    },
    'Yp1': {
        'name': 'Y position (edges)',
        'units': 'm',
        'label': 'y_edge',
        'periodic': False
    },
    'X': {
        'name': 'X position',
        'units': 'm',
        'label': 'x_centre',
        'periodic': False
    },
    'Xp1': {
        'name': 'X position (edges)',
        'units': 'm',
        'label': 'x_centre',
        'periodic': False
    },
    'Z': {
        'name': 'Height',
        'units': 'm',
        'label': 'z_centre',
        'periodic': False
    },
    'Zp1': {
        'name': 'Height (edges)',
        'units': 'm',
        'label': 'z_edge',
        'periodic': False
    },
    'Zl': {
        'name': 'Height (lower)',
        'units': 'm',
        'label': 'z_lower',
        'periodic': False
    },
    'Zu': {
        'name': 'Height (upper)',
        'units': 'm',
        'label': 'z_upper',
        'periodic': False
    },
}