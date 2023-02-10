import random
import time
from typing import Any

import cv2
import numpy as np

from poisson_util.io import read_image, write_image
from poisson_util.args import get_args
from poisson_util.io import write_image
from poisson_util.process import BaseProcessor, EquProcessor, GridProcessor


class GUI(object):
    """A simple GUI implementation.

    3 windows:

    1. src with rect bbox;
    2. tgt with single point;
    3. result with fix rate refresh.
    """

    def __init__(self, proc: BaseProcessor, src: np.ndarray, tgt: np.ndarray, out: str, n: int, scale_percent: int):
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
        self.proc = proc

        # cv2.namedWindow("source")
        # cv2.setMouseCallback("source", self.source_callback)
        # cv2.namedWindow("target")
        # cv2.setMouseCallback("target", self.target_callback)
        self.gui_src = self.src.copy()
        self.gui_tgt = self.tgt.copy()
        self.gui_out = self.tgt.copy()
        self.on_source = False
        # while True:
        #   cv2.imshow("source", self.gui_src)
        #   cv2.imshow("target", self.gui_tgt)
        #   cv2.imshow("result", self.gui_out)
        #   key = cv2.waitKey(30) & 0xFF
        #   if key == 27:
        #     break
        write_image(self.out, self.gui_out)
        # cv2.destroyAllWindows()

    def source_callback(
            self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_source = True
            self.x0, self.y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.on_source:
                self.gui_src = self.src.copy()
                cv2.rectangle(
                    self.gui_src, (self.x0, self.y0), (x, y), (255, 255, 255), 1
                )
        elif event == cv2.EVENT_LBUTTONUP:
            self.on_source = False
            self.x1, self.y1 = x, y
            self.gui_src = self.src.copy()
            cv2.rectangle(
                self.gui_src, (self.x0, self.y0), (x, y), (255, 255, 255), 1
            )
            self.x0, self.x1 = min(self.x0, self.x1), max(self.x0, self.x1)
            self.y0, self.y1 = min(self.y0, self.y1), max(self.y0, self.y1)

    def target_callback(
            self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.gui_tgt = self.tgt.copy()
            mask_x = min(self.x1 - self.x0, self.tgt.shape[1] - x)
            mask_y = min(self.y1 - self.y0, self.tgt.shape[0] - y)
            cv2.rectangle(
                self.gui_tgt,
                (x, y),
                (x + mask_x, y + mask_y),
                (255, 255, 255),
                1,
            )
            mask = np.zeros([mask_y, mask_x], np.uint8) + 255
            t = time.time()
            self.proc.reset(self.src, mask, self.tgt, (self.y0, self.x0), (y, x))
            self.gui_out, err = self.proc.step(self.n)  # type: ignore
            t = time.time() - t
            print(
                f"Time elapsed: {t:.4f}s, mask size {mask.shape}, abs Error: {err}\t"
                f"Args: -n {self.n} -h0 {self.y0} -w0 {self.x0} -h1 {y} -w1 {x}"
            )


def main() -> None:
    args = get_args("gui")

    proc: BaseProcessor
    if args.method == "equ":
        proc = EquProcessor(
            args.gradient,
            args.backend,
            args.cpu,
            args.mpi_sync_interval,
            args.block_size,
        )
    else:
        proc = GridProcessor(
            args.gradient,
            args.backend,
            args.cpu,
            args.mpi_sync_interval,
            args.block_size,
            args.grid_x,
            args.grid_y,
        )
    print(
        f"Successfully initialize PIE {args.method} solver "
        f"with {args.backend} backend"
    )

    GUI(proc, args.source, args.target, args.output, args.n)


def flatten(l):
    return [item for sublist in l for item in sublist]


def create_poisson_img(src, tgt, fst_bb, tgt_bbs, result=None):
    proc = EquProcessor(
        'max',
        'numpy',
        1
    )
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
    fst_bb_clone, gui, mask_x, mask_y, scale_percent = method_name(fst_bb, gui, increase_overlap, mask_x, mask_y, overlap_per, proc,
                                                    result, scale_percent, src, tgt, tgt_bbs, valid_points)

    choice_x, choice_y = random.choice(valid_points)
    # print(choice_x, choice_y)
    mask_on_tgt = (choice_y, choice_x)
    mask = np.zeros([mask_y, mask_x], np.uint8) + 255
    gui.proc.reset(gui.src, mask, gui.tgt, (int(fst_bb_clone[1]), int(fst_bb_clone[0])), mask_on_tgt)
    gui.gui_out, err = gui.proc.step(10000)
    return gui.gui_out, choice_x, choice_y, scale_percent


def method_name(fst_bb, gui, increase_overlap, mask_x, mask_y, overlap_per, proc, result, scale_percent, src, tgt,
                tgt_bbs, valid_points):
    while len(valid_points) == 0:
        scale_percent -= 5
        if scale_percent == 0:
            scale_percent = 50
            increase_overlap = True
            if increase_overlap >= 1:
                raise ValueError
        gui = GUI(proc, src, tgt, result, 100, scale_percent=scale_percent)
        fst_bb_clone = fst_bb * (scale_percent / 100)

        mask_x = int(fst_bb_clone[2] - fst_bb_clone[0])
        mask_y = int(fst_bb_clone[3] - fst_bb_clone[1])

        for i in range(gui.tgt.shape[1]):
            for j in range(gui.tgt.shape[0]):
                x_min_src_bb, y_min_src_bb, x_max_src_bb, y_max_src_bb = i, j, min(i + mask_x, gui.tgt.shape[1]), min(j + mask_y, gui.tgt.shape[0])
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
                        square = (min(x_max_tgt_bb, x_max_src_bb) - max(x_min_src_bb, x_min_tgt_bb)) * (y_max_src_bb - y_min_tgt_bb) / (
                                    (x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                    # check right upper point
                    if (x_max_src_bb > x_min_tgt_bb and y_min_src_bb > y_min_tgt_bb):
                        square = (x_max_src_bb - x_min_tgt_bb) * (min(y_max_src_bb, y_max_tgt_bb) - y_min_src_bb) / ((x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                    # check left upper point
                    if (x_min_src_bb > x_min_tgt_bb and y_min_src_bb > y_min_tgt_bb):
                        square = (x_max_tgt_bb - x_min_src_bb) * (min(y_max_src_bb, y_max_tgt_bb) - y_min_src_bb) / ((x_max_src_bb - x_min_src_bb) * (y_max_src_bb - y_min_src_bb))
                        intersections.append(abs(square))
                    # check left down point
                    if (x_min_src_bb > x_min_tgt_bb and y_max_src_bb > y_min_tgt_bb):
                        square = (min(x_max_tgt_bb, x_max_src_bb) - max(x_min_tgt_bb, x_min_src_bb)) * (y_max_src_bb - y_min_tgt_bb) / (
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
