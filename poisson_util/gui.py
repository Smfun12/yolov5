import time
from typing import Any

import cv2
import numpy
import numpy as np

from poisson_util.io import read_image, write_image
from poisson_util.args import get_args
from poisson_util.process import BaseProcessor, EquProcessor, GridProcessor
from utils.general import xywhn2xyxy


class GUI(object):
    """A simple GUI implementation.

    3 windows:

    1. src with rect bbox;
    2. tgt with single point;
    3. result with fix rate refresh.
    """

    def __init__(self, proc: BaseProcessor, src: numpy.ndarray, tgt: numpy.ndarray, out: str, n: int):
        super().__init__()
        self.xt, self.yt = 0, 0
        self.src = src
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


def create_poisson_img(src, tgt, fst_bb, result=None):
    proc = EquProcessor(
        'max',
        'numpy',
        1
    )
    gui = GUI(proc, src, tgt, result, 100)
    mask_x = int(fst_bb[2]-fst_bb[0])
    mask_y = int(fst_bb[3]-fst_bb[1])
    mask_x = min(mask_x, tgt.shape[1])
    mask_y = min(mask_y, tgt.shape[0])
    mask = np.zeros([mask_y, mask_x], np.uint8) + 255
    if mask.shape[0] > src.shape[0]:
        print("error")
    gui.proc.reset(gui.src, mask, gui.tgt, (0, 0), (0, 0))
    gui.gui_out, err = gui.proc.step(1000)
    return gui.gui_out
