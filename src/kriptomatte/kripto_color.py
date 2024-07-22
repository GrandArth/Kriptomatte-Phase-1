import colorsys

import numpy as np


class RandomColor:
    def __init__(self, init_sat, init_hue, init_luma = 0.07):
        self.init_hue = init_hue
        self.init_sat = init_sat
        self.init_luma = init_luma
        self.update_counter = 0

    def update_hls(self):
        if self.update_counter % 3 == 0:
            self.init_hue = (self.init_hue + 0.04) % 1
        if self.update_counter % 3 == 1:
            self.init_sat = (self.init_sat + 0.14) % 1
        if self.update_counter % 3 == 2:
            self.init_luma = (self.init_luma + 0.07) % 0.5 + 0.07
        self.update_counter = self.update_counter + 1

    def random_color(self):
        r, g, b = colorsys.hls_to_rgb(self.init_hue, self.init_luma, self.init_sat)
        rgb_256_int8 = []
        for color in [r, g, b]:
            color_in_numpy = np.array(color, dtype=np.float32)
            color_in_numpy = (np.clip(color_in_numpy * 255, 0, 255)).astype(np.uint8)
            col_list = color_in_numpy.tolist()
            rgb_256_int8.append(col_list)
        self.update_hls()
        return rgb_256_int8
