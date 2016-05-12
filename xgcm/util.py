def append_to_name(array, append):
    try:
        return array.name + "_" + append
    except TypeError:
        return append