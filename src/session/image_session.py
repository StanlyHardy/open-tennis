from tkinter import Tcl

import cv2

from glob import glob

from src.session.session_context import SessionContext


class ImageStreamer(SessionContext):

    def __init__(self):
        super().__init__()
        PATH = '/home/hardy/Workspace/Python/SportRadar/AnnotationCreator/assets/dataset/split/test/images'
        self.img_paths = glob(PATH + "/*.jpg")
        Tcl().call('lsort', '-dict', self.img_paths)

        self.total_frame_count = len(self.img_paths)

    def update_frames(self):

        frame = cv2.imread(self.img_paths[self.frame_count])
        self.set_overlay_frame(frame)
        self.set_detection_frame(frame)
        self.frame_count += 1
        if self.frame_count == self.total_frame_count:
            print("Completed!...")
            exit()

    def __del__(self):
        self._switch_off()

    def _switch_off(self):
        cv2.destroyAllWindows()
