import numpy as np


def hsv_to_hsl(hsv):
    h, sat, val = hsv
    if sat > 1:
        sat = sat/100
    if val > 1:
        val = val/100

    # hue h stays the same
    h = h

    # lightness l
    l = val - sat * val / 2

    # saturation s
    if l in (0, 1):
        s = 0
    else:
        s = (val - l) / min(l, 1-l)
    # return code as tuple
    return h, s, l


green_hsv = (240, 90, 80)
green_hsl = hsv_to_hsl(green_hsv)
print(green_hsl)
