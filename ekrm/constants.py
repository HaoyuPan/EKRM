import enum


class Channel(enum.Enum):
    rgb_red = 'Red'
    rgb_green = 'Green'
    rgb_blue = 'Blue'
    greyscale_grey = 'Lightness'
    hsv_hue = 'Hue'
    hsv_saturation = 'Saturation'
    hsv_value = 'Value'
    lab_lightness_star = 'Lightness Star'
    lab_a_star = 'A Star'
    lab_b_star = 'B Star'


class Colorspace(enum.Enum):
    rgb = 'RGB', (Channel.rgb_red, Channel.rgb_green, Channel.rgb_blue)
    hsv = 'HSV', (Channel.hsv_hue, Channel.hsv_saturation, Channel.hsv_value)
    lab = 'LAB', (Channel.lab_lightness_star, Channel.lab_a_star, Channel.lab_b_star)
    grey = 'Grey', (Channel.greyscale_grey,)

    def __init__(self, name, channels):
        self.label = name
        self.channels = channels


class SaveAs(enum.Enum):
    rgb = 'rgb'
    index = 'index'
    gray = 'gray'
    bit = 'bit'
