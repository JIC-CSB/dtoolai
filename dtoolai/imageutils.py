def coerce_to_target_dim(im, input_format):
    """Convert a PIL image to a fixed size and number of channels."""

    mode_map = {
        1: 'L',
        3: 'RGB',
        4: 'RGBA'
    }

    if im.mode not in mode_map.values():
        raise Exception(f"Unknown image mode: {im.mode}")

    ch, tdimw, tdimh = input_format
    if ch not in mode_map:
        raise Exception(f"Unsupported input format: {input_format}")

    cdimw, cdimh = im.size
    if (cdimw, cdimh) != (tdimw, tdimh):
        im = im.resize((tdimw, tdimh))

    if mode_map[ch] != im.mode:
        im = im.convert(mode_map[ch])

    return im
