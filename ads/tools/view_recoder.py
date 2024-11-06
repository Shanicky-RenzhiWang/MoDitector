import cv2
import numpy as np

class DisplayInterface(object):

    def __init__(self):
        self._width = 1200
        self._height = 400
        self._surface = None

    def run_interface(self, rgb_front, render_txt):
        # rgb_front = cv2.cvtColor(view_image[1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb_front, (1200, 400))
        surface = np.zeros((400, 1200, 3), np.uint8)
        surface[:, :1200] = rgb

        vis_text = render_txt
        surface = cv2.putText(surface, vis_text['basic_info'], (20, 290), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                              (0, 0, 255), 1)
        # surface = cv2.putText(surface, vis_text['plan'], (20, 710), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        surface = cv2.putText(surface, vis_text['control'], (20, 330), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                              (0, 0, 255),
                              1)
        surface = cv2.putText(surface, vis_text['repair'], (20, 370), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                              (0, 0, 255),
                              1)
        return surface
