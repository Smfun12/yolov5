import random

import PIL
import cv2
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from poisson_util.io import write_image


class GUI(object):
    """A simple GUI implementation.

    3 windows:

    1. src with rect bbox;
    2. tgt with single point;
    3. result with fix rate refresh.
    """

    def __init__(self, src: np.ndarray, tgt: np.ndarray, out: str, n: int, scale_percent: int):
        super().__init__()
        self.xt, self.yt = 0, 0
        # self.src = read_image(src)
        self.src = src
        width = int(self.src.shape[1] * scale_percent / 100)
        height = int(self.src.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        self.src = cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)
        # self.tgt = read_image(tgt)
        self.tgt = tgt
        self.x0, self.y0 = 0, 0
        self.y1, self.x1 = self.src.shape[:2]
        self.out = out
        self.n = n

        self.gui_src = self.src.copy()
        self.gui_tgt = self.tgt.copy()
        self.gui_out = self.tgt.copy()
        self.on_source = False
        write_image(self.out, self.gui_out)


def flatten(l):
    return [item for sublist in l for item in sublist]


def naive_copy_paste_img(src, tgt, fst_bb, tgt_bbs, result=None):
    exclude_xx = []
    exclude_yy = []
    for tgt_bb in tgt_bbs:
        x_min, x_max = int(tgt_bb[0]), int(tgt_bb[2])
        y_min, y_max = int(tgt_bb[1]), int(tgt_bb[3])
        exclude_xx.append((x_min, x_max))
        exclude_yy.append((y_min, y_max))
    valid_points = []
    gui = None
    mask_x, mask_y = None, None
    increase_overlap = False
    scale_percent = 105
    overlap_per = 0.05
    padding = 10
    fst_bb_clone, gui, _, _, scale_percent = paste_object(fst_bb, gui, increase_overlap, mask_x, mask_y,
                                                                    overlap_per,
                                                                    result, scale_percent, src, tgt, tgt_bbs,
                                                                    valid_points, padding)
    # Points on tgt
    choice_x, choice_y = random.choice(valid_points)

    bbox_x0 = max(0, int(fst_bb_clone[0])-2*padding)
    bbox_y0 = max(0, int(fst_bb_clone[1])-2*padding)
    bbox_x1 = min(gui.src.shape[1], int(fst_bb_clone[2])+2*padding)
    bbox_y1 = min(gui.src.shape[0], int(fst_bb_clone[3])+2*padding)
    width, height = choice_x + int(bbox_x1 - bbox_x0), choice_y + int(bbox_y1 - bbox_y0)

    # Src
    src_pil = Image.fromarray(gui.src)
    # draw_mask_on_src = ImageDraw.Draw(src_pil)
    # draw_mask_on_src.rectangle(((bbox_x0, bbox_y0), (bbox_x1, bbox_y1)))
    src_pil_copy = src_pil.copy()
    src_pil_copy = src_pil_copy.crop((bbox_x0, bbox_y0, bbox_x1, bbox_y1))
    # Mask
    mask_im = Image.new("L", src_pil.size, 0)
    draw_mask = ImageDraw.Draw(mask_im)
    draw_mask.rectangle(((bbox_x0+padding, bbox_y0+padding), (bbox_x1-padding, bbox_y1-padding)), fill=255)
    mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(5))
    mask_im_blur_copy = mask_im_blur.copy()
    mask_im_blur_copy = mask_im_blur_copy.crop((bbox_x0, bbox_y0, bbox_x1, bbox_y1))
    # Tgt
    tgt_pil = Image.fromarray(tgt)
    # draw_mask_on_tgt = ImageDraw.Draw(tgt_pil)
    # draw_mask_on_tgt.rectangle(((choice_x, choice_y), (width, height)))
    tgt_copy = tgt_pil.copy()
    tgt_copy.paste(src_pil_copy, (choice_x, choice_y, width, height), mask_im_blur_copy)

    # Convert Pil to cv2
    tgt = np.array(tgt_copy)

    return tgt, choice_x+padding, choice_y+padding, scale_percent


def paste_object(fst_bb, gui, increase_overlap, mask_x, mask_y, overlap_per, result, scale_percent, src, tgt,
                 tgt_bbs, valid_points, padding):
    while len(valid_points) == 0:
        scale_percent -= 5
        if scale_percent < 50:
            scale_percent = 50
            increase_overlap = True
            if increase_overlap >= 1:
                raise ValueError
        gui = GUI(src, tgt, result, 100, scale_percent=scale_percent)
        fst_bb_clone = fst_bb * (scale_percent / 100)

        mask_x = int(fst_bb_clone[2] - fst_bb_clone[0])
        mask_y = int(fst_bb_clone[3] - fst_bb_clone[1])

        for i in range(gui.tgt.shape[1]):
            for j in range(gui.tgt.shape[0]):
                x_min_src_bb, y_min_src_bb, x_max_src_bb, y_max_src_bb = max(0, i-padding), max(0,j-padding), min(i + mask_x+padding, gui.tgt.shape[1]), min(
                    j + mask_y+padding, gui.tgt.shape[0])
                intersections = []
                can_add = True
                for tgt_bb in tgt_bbs:
                    x_min_tgt_bb, x_max_tgt_bb = int(tgt_bb[0]), int(tgt_bb[2])
                    y_min_tgt_bb, y_max_tgt_bb = int(tgt_bb[1]), int(tgt_bb[3])
                    # check if it is located in tgt bb
                    if x_min_src_bb >= x_min_tgt_bb and y_min_src_bb >= y_min_tgt_bb and x_max_src_bb <= x_max_tgt_bb and y_max_src_bb <= y_max_tgt_bb:
                        can_add = False
                        break
                    if x_min_tgt_bb >= x_min_tgt_bb and y_min_tgt_bb >= y_min_src_bb and x_max_src_bb >= x_max_src_bb and y_max_src_bb >= y_max_tgt_bb:
                        can_add = False
                        break
                    # check right down point
                    if (x_max_src_bb > x_min_tgt_bb and y_max_src_bb > y_min_tgt_bb):
                        square = (min(x_max_tgt_bb, x_max_src_bb) - max(x_min_src_bb, x_min_tgt_bb)) * (
                                    y_max_src_bb - y_min_tgt_bb) / (
                                         (x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                    # check right upper point
                    if (x_max_src_bb > x_min_tgt_bb and y_min_src_bb > y_min_tgt_bb):
                        square = (x_max_src_bb - x_min_tgt_bb) * (min(y_max_src_bb, y_max_tgt_bb) - y_min_src_bb) / (
                                    (x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                    # check left upper point
                    if (x_min_src_bb > x_min_tgt_bb and y_min_src_bb > y_min_tgt_bb):
                        square = (x_max_tgt_bb - x_min_src_bb) * (min(y_max_src_bb, y_max_tgt_bb) - y_min_src_bb) / (
                                    (x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                    # check left down point
                    if (x_min_src_bb > x_min_tgt_bb and y_max_src_bb > y_min_tgt_bb):
                        square = (min(x_max_tgt_bb, x_max_src_bb) - max(x_min_tgt_bb, x_min_src_bb)) * (
                                    y_max_src_bb - y_min_tgt_bb) / (
                                         (x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                if increase_overlap:
                    overlap_per += min(1.0, increase_overlap + 0.05)
                    increase_overlap = False
                if sum(intersections) / max(1, len(intersections)) >= overlap_per:
                    can_add = False
                if can_add and (i + mask_x < gui.tgt.shape[1]) and (j + mask_y < gui.tgt.shape[0]):
                    valid_points.append((i, j))
    print("Scale_percent={}, overlap_per={}".format(scale_percent, overlap_per))
    return fst_bb_clone, gui, mask_x, mask_y, scale_percent
